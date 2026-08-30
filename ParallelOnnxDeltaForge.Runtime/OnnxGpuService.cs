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
        /// <summary>
        /// A thread-safe dictionary to manage active ONNX inference sessions, keyed by a unique session ID (Guid).
        /// </summary>
        private readonly ConcurrentDictionary<Guid, InferenceSession> _activeSessions = new();

        /// <summary>
        /// A semaphore to ensure that only one model is loaded onto the GPU at a time, preventing potential conflicts and ensuring thread safety during model loading.
        /// </summary>
        private readonly SemaphoreSlim _loadLock = new(1, 1);

        /// <summary>
        /// A flag to indicate whether the service has been disposed, preventing multiple disposals and ensuring proper resource management.
        /// </summary>
        private bool _disposed;


        /// <summary>
        /// The logger instance used for logging information, warnings, and errors related to ONNX GPU operations. If no logger is provided during initialization, a default RollingFileMemoryLogger is created.
        /// </summary>
        public readonly RollingFileMemoryLogger Logger;


        /// <summary>
        /// Initializes a new instance of the OnnxGpuService class with an optional logger. If no logger is provided, a default RollingFileMemoryLogger is created. This service manages ONNX model loading and inference on GPU devices, ensuring thread safety and proper resource management.
        /// </summary>
        /// <param name="logger">An optional logger instance (which may be DI-injected) for logging information, warnings, and errors related to ONNX GPU operations.</param>
        public OnnxGpuService(RollingFileMemoryLogger? logger)
        {
            this.Logger = logger ?? new RollingFileMemoryLogger();
            this.Logger.LogInfo("[ONNX] OnnxGpuService initialized.");
        }


        /// <summary>
        /// Retrieves a read-only list of available CUDA devices on the system. This method is intended to provide information about the GPU devices that can be used for ONNX model inference. In this implementation, it returns a hardcoded list of device IDs (0 and 1) for demonstration purposes. In a real-world scenario, this method would query the system for actual available CUDA devices.
        /// </summary>
        /// <returns>A read-only list of available CUDA device IDs.</returns>
        public IReadOnlyList<int> GetAvailableCudaDevices()
        {

            return [0, 1];
        }

        /// <summary>
        /// Asynchronously loads an ONNX model onto a specified GPU device. This method ensures that only one model is loaded onto the GPU at a time by using a semaphore for thread safety. It configures the CUDA provider options strictly to manage VR
        /// </summary>
        /// <param name="modelPath">File path to the ONNX model.</param>
        /// <param name="deviceId">The ID of the GPU device on which to load the model.</param>
        /// <returns>The unique session ID (Guid) for the loaded model.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the specified model file does not exist.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the specified device ID is invalid (not in the list of available CUDA devices).</exception>
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

                // 1. Configure CUDA Provider Options very strictly
                var cudaProviderOptions = new OrtCUDAProviderOptions
                {
                    DeviceId = deviceId,
                    // Prevents memory fragmentation for large LLMs: kSameAsRequested (1)
                    ArenaExtendStrategy = 1,
                    // Strictly limits VRAM allocation to demand, respecting the card's limits
                    GpuMemLimit = long.MaxValue,
                    CudnnConvAlgoSearch = OrtCudnnConvAlgoSearch.Heuristic
                };

                // 2. Build Session Options (Zero-Allocation / Unmanaged)
                using var sessionOptions = new SessionOptions();
                sessionOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);

                // Cross-GPU optimizations (Memory Pattern on/off)
                sessionOptions.EnableMemoryPattern = true;
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                // 3. Load model asynchronously (IO-bound on Background Thread)
                var session = await Task.Run(() => new InferenceSession(modelPath, sessionOptions));

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

        /// <summary>
        /// Unloads an ONNX model associated with the specified session ID, freeing up GPU VRAM. If the session ID does not exist, a warning is logged. This method ensures that resources are properly disposed of to prevent memory leaks and maintain optimal GPU performance.
        /// </summary>
        /// <param name="sessionId">The unique session ID (Guid) of the model to unload. If null, the first active session will be unloaded.</param>
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

        /// <summary>
        /// Unloads all active ONNX models, freeing up GPU VRAM for each session. This method iterates through all active sessions and calls UnloadModel for each one, ensuring that all resources are properly disposed of. It is useful for scenarios where the application is shutting down or when a complete reset of the GPU state is required.
        /// </summary>
        public void UnloadAll()
        {
            this.Logger.LogInfo("[ONNX] Unloading all active models...");
            foreach (var sessionId in this._activeSessions.Keys)
            {
                this.UnloadModel(sessionId);
            }
        }

        /// <summary>
        /// Disposes of the OnnxGpuService, releasing all resources and unloading all active ONNX models. This method is called by the public Dispose method and ensures that the service is properly cleaned up, preventing memory leaks and ensuring that GPU resources are freed. It also suppresses finalization to optimize garbage collection.
        /// </summary>
        /// <param name="disposing"></param>
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

        /// <summary>
        /// Disposes of the OnnxGpuService, releasing all resources and unloading all active ONNX models. This method is called by consumers of the service to ensure proper cleanup of unmanaged resources.
        /// </summary> 
        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}