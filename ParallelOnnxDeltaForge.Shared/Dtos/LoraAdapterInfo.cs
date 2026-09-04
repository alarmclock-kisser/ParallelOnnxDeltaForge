using System;
using System.Collections.Generic;

namespace ParallelOnnxDeltaForge.Shared.Dtos
{
    /// <summary>
    /// Metadata for a loaded LoRA adapter, including its configuration and layer mappings.
    /// </summary>
    public sealed class LoraAdapterInfo
    {
        /// <summary>
        /// Unique identifier for the adapter.
        /// </summary>
        public Guid Id { get; set; } = Guid.NewGuid();

        /// <summary>
        /// File path from which the adapter was loaded.
        /// </summary>
        public string AdapterPath { get; set; } = string.Empty;

        /// <summary>
        /// Name or label for the adapter.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// LoRA rank (r) – the dimensionality of the low-rank decomposition.
        /// </summary>
        public int Rank { get; set; }

        /// <summary>
        /// Global scaling factor applied to the LoRA delta (α / r conventionally).
        /// </summary>
        public float ScaleFactor { get; set; } = 1f;

        /// <summary>
        /// Names of model layers this adapter targets (e.g., "encoder.block.0.self_attention.q_proj").
        /// </summary>
        public IReadOnlyList<string> TargetLayers { get; set; } = Array.Empty<string>();
    }
}
