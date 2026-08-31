using ParallelOnnxDeltaForge.Shared.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace ParallelOnnxDeltaForge.Shared.MediaDtos
{
    /// <summary>
    /// Image data DTO that encapsulates image information and the corresponding image data.
    /// </summary>
    public class ImageData : IMediaData
    {
        public required IMediaInfo Info { get; set; }

        public string? Pointer => this.Info.Pointer;

        public bool OnGpu => !string.IsNullOrEmpty(this.Pointer) && !this.Pointer.Equals("null", StringComparison.OrdinalIgnoreCase) && !this.Pointer.Equals(IntPtr.Zero.ToString(), StringComparison.OrdinalIgnoreCase);

        public string MimeType { get; set; } = "image/png";

        public string Base64Data { get; set; } = string.Empty;

        public string Base64Image => $"data:{this.MimeType};base64,{this.Base64Data}";

        public float DataSizeMb => this.Base64Data.LongCount() * 4f / 3f / 1024f / 1024f;

        public bool IdMatch(string id, bool requireOnGpu = false)
        {
            return this.Info.IdMatch(id, requireOnGpu);
        }
    }
}
