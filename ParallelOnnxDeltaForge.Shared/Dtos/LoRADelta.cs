using System;
using System.Collections.Generic;

namespace ParallelOnnxDeltaForge.Shared.Dtos
{
    /// <summary>
    /// Represents a single LoRA weight delta for one layer: the low-rank matrices A and B
    /// such that the effective weight change is ΔW = B × A.
    /// </summary>
    public sealed class LoRADelta
    {
        /// <summary>
        /// Layer name this delta applies to.
        /// </summary>
        public string LayerName { get; set; } = string.Empty;

        /// <summary>
        /// Shape of the LoRA A matrix: [original_dim, rank].
        /// </summary>
        public long[]? AShape { get; set; }

        /// <summary>
        /// Flat float array for matrix A (row-major).
        /// </summary>
        public float[]? AData { get; set; }

        /// <summary>
        /// Shape of the LoRA B matrix: [rank, original_dim].
        /// </summary>
        public long[]? BShape { get; set; }

        /// <summary>
        /// Flat float array for matrix B (row-major).
        /// </summary>
        public float[]? BData { get; set; }

        /// <summary>
        /// Scaling factor for this layer's delta.
        /// </summary>
        public float ScaleFactor { get; set; } = 1f;
    }

    /// <summary>
    /// A collection of LoRA deltas keyed by layer name, representing the full accumulated delta for an adapter.
    /// </summary>
    public sealed class LoRADeltaSet
    {
        /// <summary>
        /// Unique identifier for this delta set.
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Human-readable label.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// LoRA rank used for decomposition.
        /// </summary>
        public int Rank { get; set; }

        /// <summary>
        /// Per-layer deltas.
        /// </summary>
        public IReadOnlyDictionary<string, LoRADelta> Deltas { get; set; } = new Dictionary<string, LoRADelta>();

        /// <summary>
        /// Number of chat turns the deltas were accumulated over.
        /// </summary>
        public int AccumulatedTurns { get; set; }
    }
}
