using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using ParallelOnnxDeltaForge.Shared.Dtos;

namespace ParallelOnnxDeltaForge.Shared.Interfaces
{
    public interface IOnnxDeltaForgeService : IDisposable
    {
        Task<Guid> LoadModelAsync(string modelPath, int deviceId);
        Task<LoraAdapterInfo> LoadLoraAdapterAsync(string adapterPath, string name, int rank, float scaleFactor);
        Task<Guid> LoadModelWithLoraAsync(string modelPath, string loraPath, string loraName, int rank, float scaleFactor, int deviceId);
        Task<InferenceResponse> RunInferenceAsync(InferenceRequest request);
        Task<LoRADeltaSet> ComputeDeltasAsync(int targetRank);
        Task<DeltaExportResult> ExportDeltasAsync(LoRADeltaSet deltaSet, DeltaExportMode mode, string outputPath);
        Task ClearContextAsync();
        IReadOnlyList<ContextTurn> GetContext();
        IReadOnlyList<LoraAdapterInfo> GetLoadedAdapters();
        void UnloadModel(Guid? sessionId);
        void UnloadAll();
    }

    public interface ILoRAAdapterLoader
    {
        Task<LoraAdapterInfo> LoadAsync(string adapterPath, string name, int rank, float scaleFactor);
        IReadOnlyDictionary<string, LoraAdapterInfo> GetLoadedAdapters();
    }

    public interface IContextTracker
    {
        void RecordTurn(ContextTurn turn);
        IReadOnlyList<ContextTurn> GetTurns();
        void Clear();
        int TurnCount { get; }
    }

    public interface IDeltaComputationService
    {
        Task<LoRADeltaSet> ComputeFromContextAsync(IReadOnlyList<ContextTurn> turns, int targetRank);
    }

    public interface IDeltaExporter
    {
        Task<DeltaExportResult> ExportAsLoraAdapterAsync(LoRADeltaSet deltaSet, string outputPath);
        Task<DeltaExportResult> MergeIntoBaseModelAsync(LoRADeltaSet deltaSet, string baseModelPath, string outputPath);
    }
}
