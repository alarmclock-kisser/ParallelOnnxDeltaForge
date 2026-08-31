using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using ParallelOnnxDeltaForge.Shared.Interfaces;
using ParallelOnnxDeltaForge.Shared;

namespace ParallelOnnxDeltaForge.Runtime
{
    public class OnnxGpuService : IOnnxGpuService
    {
        private readonly ConcurrentDictionary<Guid, InferenceSession> _activeSessions = new();
        private readonly SemaphoreSlim _loadLock = new(1, 1);
        private bool _disposed;

        public readonly RollingFileMemoryLogger Logger;

        public OnnxGpuService(RollingFileMemoryLogger? logger)
        {
            this.Logger = logger ?? new RollingFileMemoryLogger();
            this.Logger.LogInfo("[ONNX] OnnxGpuService initialized.");
        }


        public IReadOnlyList<int> GetAvailableCudaDevices()
        {

            return [0, 1];
        }

        public async Task<Guid> LoadModelAsync(string modelPath, int deviceId)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }
            if (deviceId < 0 || deviceId >= this.GetAvailableCudaDevices().Count)
            {
                throw new ArgumentOutOfRangeException(nameof(deviceId), $"Invalid GPU device ID: {deviceId}");
            }

            Guid sessionId = Guid.NewGuid();

            // Prevents two models from being loaded onto the GPU simultaneously over the PCIe bus
            await this._loadLock.WaitAsync();
            try
            {
                this.Logger.LogInfo($"[ONNX] Loading model '{Path.GetFileName(modelPath)}' onto GPU {deviceId}...");

                // 1. Configure CUDA Provider Options using the modern Dictionary API
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                using var sessionOptions = new SessionOptions();
                InferenceSession session;
                try
                {
                    cudaProviderOptions.UpdateOptions(new Dictionary<string, string>
                {
                    { "device_id", deviceId.ToString() },
                    // Prevents memory fragmentation for large LLMs: kSameAsRequested (1)
                    { "arena_extend_strategy", "kSameAsRequested" },
                    // Strictly limits VRAM allocation to demand, respecting the card's limits
                    { "gpu_mem_limit", long.MaxValue.ToString() },
                    // Heuristic search for cuDNN convolution algorithms
                    { "cudnn_conv_algo_search", "HEURISTIC" },
                    // Optimizes memory copies between host and device
                    { "do_copy_in_default_stream", "1" }
                });

                    // 2. Append CUDA Execution Provider to Session Options
                    sessionOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);

                    // Cross-GPU optimizations (Memory Pattern on/off)
                    sessionOptions.EnableMemoryPattern = true;
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                    // 3. Load model asynchronously (IO-bound on Background Thread)
                    session = await Task.Run(() => new InferenceSession(modelPath, sessionOptions));
                }
                catch (OnnxRuntimeException onnxEx)
                {
                    this.Logger.LogError($"[ONNX] ONNX Runtime error while loading model: {onnxEx.Message}");
                    throw;
                }

                if (!this._activeSessions.TryAdd(sessionId, session))
                {
                    session.Dispose();
                    throw new InvalidOperationException("Failed to register the inference session.");
                }

                this.Logger.LogSuccess($"[ONNX] Model loaded successfully on GPU {deviceId}. SessionID: {sessionId}");
                return sessionId;
            }
            catch (Exception ex)
            {
                this.Logger.LogError($"[ONNX] Error loading model on GPU {deviceId}: {ex.Message}");
                throw;
            }
            finally
            {
                this._loadLock.Release();
            }
        }

        public void UnloadModel(Guid? sessionId)
        {
            sessionId ??= this._activeSessions.Keys.FirstOrDefault();
            if (this._activeSessions.TryRemove(sessionId.Value, out var session))
            {
                session.Dispose();
                this.Logger.LogInfo($"[ONNX] Session {sessionId} unloaded and VRAM freed.");
            }
            else
            {
                this.Logger.LogWarning($"[ONNX] Attempted to unload non-existent session {sessionId}.");
            }
        }

        public void UnloadAll()
        {
            this.Logger.LogInfo("[ONNX] Unloading all active models...");
            foreach (var sessionId in this._activeSessions.Keys)
            {
                this.UnloadModel(sessionId);
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!this._disposed)
            {
                if (disposing)
                {
                    this.UnloadAll();
                    this._loadLock.Dispose();
                }
                this._disposed = true;
            }
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}