using System;
using System.Collections.Generic;

namespace ParallelOnnxDeltaForge.Shared.Dtos
{
    /// <summary>
    /// Represents a single chat turn captured during a session, used for LoRA delta computation.
    /// </summary>
    public sealed class ContextTurn
    {
        /// <summary>
        /// Zero-based index of this turn within the session.
        /// </summary>
        public int TurnIndex { get; set; }

        /// <summary>
        /// The input prompt or message for this turn.
        /// </summary>
        public string Input { get; set; } = string.Empty;

        /// <summary>
        /// The model's output tokens (as text) for this turn.
        /// </summary>
        public string Output { get; set; } = string.Empty;

        /// <summary>
        /// Input tensor data fed into the model (token IDs, embeddings, etc.).
        /// </summary>
        public float[]? InputData { get; set; }

        /// <summary>
        /// Logits or output activations from the base model (before LoRA).
        /// </summary>
        public float[]? BaseOutputData { get; set; }

        /// <summary>
        /// Logits or output activations from the LoRA-adapted model.
        /// </summary>
        public float[]? LoraOutputData { get; set; }

        /// <summary>
        /// Timestamp when this turn was recorded.
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }
}
