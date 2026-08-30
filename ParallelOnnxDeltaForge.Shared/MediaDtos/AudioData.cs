using ParallelOnnxDeltaForge.Shared.Interfaces;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace ParallelOnnxDeltaForge.Shared.MediaDtos
{
    /// <summary>
    /// Audio data DTO that encapsulates audio information and the corresponding audio data.
    /// </summary>
    public class AudioData
    {
        public required IMediaInfo Info { get; set; }

        public string? Pointer => this.Info?.Pointer;

        public float[] AudioDataFloats { get; set; } = [];
        public float[][] AudioDataFloatChunks { get; set; } = [];

        public int ChunkSize => this.AudioDataFloatChunks.Length != 0 ? this.AudioDataFloatChunks.FirstOrDefault()?.Length ?? 0 : 0;
        public float DataSizeMb => this.AudioDataFloats.Length != 0 ? this.AudioDataFloats.LongCount() * sizeof(Single) / 1024f / 1024f : this.AudioDataFloatChunks.Length != 0 ? this.AudioDataFloatChunks.Sum(chunk => chunk.Length) * sizeof(Single) / 1024f / 1024f : 0f;


        public bool IdMatch(string id, bool requireOnGpu = false)
        {
            return this.Info.IdMatch(id, requireOnGpu);
        }
    }
}
