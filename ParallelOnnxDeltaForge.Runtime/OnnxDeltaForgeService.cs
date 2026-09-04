using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;
using ParallelOnnxDeltaForge.Shared.Interfaces;

namespace ParallelOnnxDeltaForge.Runtime
{
    /// <summary>
    /// Orchestrates the full LoRA delta workflow: model load, inference, context tracking,
    /// delta computation, and export (standalone adapter or merged model).
    /// </summary>
    public class OnnxDeltaForgeService : ParallelOnnxDeltaForge.Shared.Interfaces.IOnnxDeltaForgeService, IDisposable
    {
        private readonly IOnnxGpuService _gpuService;
        private readonly LoRAAdapterLoader _loraLoader;
        private readonly ContextTracker _contextTracker;
        private readonly LoRADeltaComputationService _deltaComputation;
        private readonly DeltaExporter _deltaExporter;
        private readonly RollingFileMemoryLogger _logger;
        private readonly SemaphoreSlim _inferenceLock = new(1, 1);
        private bool _disposed;

        private Guid? _currentSessionId;
        private string? _currentModelPath;

        public OnnxDeltaForgeService(
            IOnnxGpuService gpuService,
            LoRAAdapterLoader loraLoader,
            ContextTracker contextTracker,
            LoRADeltaComputationService deltaComputation,
            DeltaExporter deltaExporter,
            RollingFileMemoryLogger logger)
        {
            this._gpuService = gpuService;
            this._loraLoader = loraLoader;
            this._contextTracker = contextTracker;
            this._deltaComputation = deltaComputation;
            this._deltaExporter = deltaExporter;
            this._logger = logger;
        }

        public async Task<Guid> LoadModelAsync(string modelPath, int deviceId)
        {
            this._logger.LogInfo($"[DeltaForge] Loading model from {modelPath} on GPU {deviceId}");
            this._currentModelPath = modelPath;
            this._currentSessionId = await this._gpuService.LoadModelAsync(modelPath, deviceId);
            return this._currentSessionId.Value;
        }

        public async Task<LoraAdapterInfo> LoadLoraAdapterAsync(string adapterPath, string name, int rank, float scaleFactor)
        {
            return await this._loraLoader.LoadAsync(adapterPath, name, rank, scaleFactor);
        }

        public async Task<Guid> LoadModelWithLoraAsync(string modelPath, string loraPath, string loraName, int rank, float scaleFactor, int deviceId)
        {
            await this.LoadModelAsync(modelPath, deviceId);
            await this.LoadLoraAdapterAsync(loraPath, loraName, rank, scaleFactor);
            this._logger.LogSuccess($"[DeltaForge] Model + LoRA '{loraName}' loaded");
            return this._currentSessionId!.Value;
        }

        public async Task<InferenceResponse> RunInferenceAsync(InferenceRequest request)
        {
            await this._inferenceLock.WaitAsync();
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                bool track = request.TrackForDelta;
                int turnIndex = this._contextTracker.TurnCount;

                // For now: simple passthrough response
                // In production this would call the actual ONNX session with tokenization
                var response = new InferenceResponse
                {
                    Output = "[Inference response]",
                    WasTracked = track,
                    TurnIndex = turnIndex,
                };

                if (track)
                {
                    var input = request.InputData ?? this.TokenizePlaceholder(request.Input);
                    this._contextTracker.RecordTurn(new ContextTurn
                    {
                        Input = request.Input,
                        InputData = input,
                    });
                }

                sw.Stop();
                response.DurationMs = sw.ElapsedMilliseconds;

                return response;
            }
            finally
            {
                this._inferenceLock.Release();
            }
        }

        public async Task<LoRADeltaSet> ComputeDeltasAsync(int targetRank)
        {
            var turns = this._contextTracker.GetTurns();
            this._logger.LogInfo($"[DeltaForge] Computing deltas from {turns.Count} turns, rank={targetRank}");
            return await this._deltaComputation.ComputeFromContextAsync(turns, targetRank);
        }

        public async Task<DeltaExportResult> ExportDeltasAsync(LoRADeltaSet deltaSet, DeltaExportMode mode, string outputPath)
        {
            if (mode == DeltaExportMode.StandaloneAdapter)
            {
                return await this._deltaExporter.ExportAsLoraAdapterAsync(deltaSet, outputPath);
            }
            else if (this._currentModelPath != null)
            {
                return await this._deltaExporter.MergeIntoBaseModelAsync(deltaSet, this._currentModelPath, outputPath);
            }
            else
            {
                throw new InvalidOperationException("No base model loaded for merge. Load a model first or use StandaloneAdapter mode.");
            }
        }

        public async Task ClearContextAsync()
        {
            this._contextTracker.Clear();
            await Task.CompletedTask;
        }

        public IReadOnlyList<ContextTurn> GetContext() => this._contextTracker.GetTurns();

        public IReadOnlyList<LoraAdapterInfo> GetLoadedAdapters()
        {
            return this._loraLoader.GetLoadedAdapters().Values.ToList().AsReadOnly();
        }

        public void UnloadModel(Guid? sessionId)
        {
            this._gpuService.UnloadModel(sessionId);
            this._currentSessionId = null;
        }

        public void UnloadAll()
        {
            this._gpuService.UnloadAll();
            this._currentSessionId = null;
        }

        private float[] TokenizePlaceholder(string text)
        {
            // Placeholder: map characters to float values
            // In production, use a real tokenizer
            return text.Select(c => (float)c).ToArray();
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!this._disposed)
            {
                if (disposing)
                {
                    this._inferenceLock.Dispose();
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
