using System;

namespace ParallelOnnxDeltaForge.Shared.Dtos
{
    /// <summary>
    /// How the LoRA deltas should be persisted.
    /// </summary>
    public enum DeltaExportMode
    {
        /// <summary>
        /// Write deltas as a standalone LoRA adapter file (.onnx).
        /// </summary>
        StandaloneAdapter = 0,

        /// <summary>
        /// Merge deltas into the base model weights and overwrite/create a new model file.
        /// </summary>
        MergeIntoModel = 1
    }

    /// <summary>
    /// Result of a delta export operation.
    /// </summary>
    public sealed class DeltaExportResult
    {
        /// <summary>
        /// Whether the export succeeded.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Path to the exported file (adapter or merged model).
        /// </summary>
        public string? OutputPath { get; set; }

        /// <summary>
        /// Error message if the export failed.
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Mode that was used for export.
        /// </summary>
        public DeltaExportMode Mode { get; set; }

        /// <summary>
        /// Total number of bytes written.
        /// </summary>
        public long BytesWritten { get; set; }

        /// <summary>
        /// Timestamp of the export.
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Request payload for running inference within a tracked session.
    /// </summary>
    public sealed class InferenceRequest
    {
        /// <summary>
        /// The input text or tokenized data for the model.
        /// </summary>
        public string Input { get; set; } = string.Empty;

        /// <summary>
        /// Whether to capture this turn for delta computation.
        /// </summary>
        public bool TrackForDelta { get; set; } = true;

        /// <summary>
        /// Optional raw input tensor data (if already tokenized).
        /// </summary>
        public float[]? InputData { get; set; }
    }

    /// <summary>
    /// Response from an inference call within a tracked session.
    /// </summary>
    public sealed class InferenceResponse
    {
        /// <summary>
        /// The model's output text or token sequence.
        /// </summary>
        public string Output { get; set; } = string.Empty;

        /// <summary>
        /// Output tensor data (logits, embeddings, etc.).
        /// </summary>
        public float[]? OutputData { get; set; }

        /// <summary>
        /// Whether the turn was tracked for delta computation.
        /// </summary>
        public bool WasTracked { get; set; }

        /// <summary>
        /// Turn index within the current session.
        /// </summary>
        public int TurnIndex { get; set; }

        /// <summary>
        /// Inference duration in milliseconds.
        /// </summary>
        public double DurationMs { get; set; }
    }
}
