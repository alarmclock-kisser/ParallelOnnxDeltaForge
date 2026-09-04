using System;

namespace ParallelOnnxDeltaForge.Shared.Dtos
{
    public sealed class LoadModelRequest
    {
        public string ModelPath { get; set; } = string.Empty;
        public int DeviceId { get; set; }
    }

    public sealed class LoadModelLoraRequest
    {
        public string ModelPath { get; set; } = string.Empty;
        public string LoraPath { get; set; } = string.Empty;
        public string LoraName { get; set; } = string.Empty;
        public int Rank { get; set; } = 8;
        public float ScaleFactor { get; set; } = 1f;
        public int DeviceId { get; set; }
    }

    public sealed class LoadLoraRequest
    {
        public string AdapterPath { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public int Rank { get; set; } = 8;
        public float ScaleFactor { get; set; } = 1f;
    }

    public sealed class ComputeDeltasRequest
    {
        public int TargetRank { get; set; } = 8;
    }

    public sealed class ExportDeltasRequest
    {
        public int TargetRank { get; set; } = 8;
        public DeltaExportMode Mode { get; set; }
        public string OutputPath { get; set; } = "deltas.onnx";
    }
}
