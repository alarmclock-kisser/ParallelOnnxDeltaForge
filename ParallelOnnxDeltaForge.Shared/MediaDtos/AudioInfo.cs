using ParallelOnnxDeltaForge.Shared.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace ParallelOnnxDeltaForge.Shared.MediaDtos
{
    /// <summary>
    /// Audiometadaten.
    /// </summary>
    public class AudioInfo : IMediaInfo
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AudioInfo"/> class.
        /// </summary>
        public AudioInfo()
        {
        }

        /// <summary>
        /// Gets or sets the unique identifier for this audio info.
        /// </summary>
        public Guid Id { get; set; }

        /// <summary>
        /// Gets or sets the creation timestamp of the audio file.
        /// </summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Gets or sets the name of the audio file.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file path of the audio file.
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the media type. Default is "audio".
        /// </summary>
        public string MediaType { get; set; } = "audio";

        /// <summary>
        /// Gets or sets the audio sample rate in Hertz.
        /// </summary>
        public int SampleRate { get; set; }

        /// <summary>
        /// Gets or sets the number of audio channels.
        /// </summary>
        public int Channels { get; set; }

        /// <summary>
        /// Gets or sets the bit depth of the audio.
        /// </summary>
        public int BitDepth { get; set; }

        /// <summary>
        /// Gets or sets the length/duration label of the audio.
        /// </summary>
        public string Length { get; set; } = "0";

        /// <summary>
        /// Gets or sets the duration in seconds.
        /// </summary>
        public float DurationSeconds { get; set; }

        /// <summary>
        /// Gets or sets the beats per minute, if available.
        /// </summary>
        public float? Bpm { get; set; } = null;

        /// <summary>
        /// Gets or sets the pointer associated with the audio, used for GPU memory management.
        /// </summary>
        public string? Pointer { get; set; } = null;

        /// <summary>
        /// Gets the chunk size of the audio data.
        /// </summary>
        public int ChunkSize { get; set; } = 0;

        /// <summary>
        /// Gets or sets the overlap factor for audio processing.
        /// </summary>
        public float Overlap { get; set; } = 0.5f;

        /// <summary>
        /// Determines if the audio is on GPU based on the pointer value.
        /// </summary>
        public bool OnGpu => !string.IsNullOrEmpty(this.Pointer) && !this.Pointer.Equals("null", StringComparison.OrdinalIgnoreCase) && !this.Pointer.Equals(IntPtr.Zero.ToString(), StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Checks if the audio ID matches the provided ID, with optional GPU requirement.
        /// </summary>
        /// <param name="id">The ID to match against.</param>
        /// <param name="requireOnGpu">Whether to require the audio to be on GPU.</param>
        /// <returns>True if the IDs match and GPU requirement is satisfied.</returns>
        public bool IdMatch(string id, bool requireOnGpu = false)
        {
            return this.Id.ToString().Equals(id, StringComparison.OrdinalIgnoreCase) && (requireOnGpu ? this.OnGpu : true);
        }

        /// <summary>
        /// Checks if the audio ID matches the provided GUI ID, with optional GPU requirement.
        /// </summary>
        /// <param name="id">The GUID to match against.</param>
        /// <param name="requireOnGpu">Whether to require the audio to be on GPU.</param>
        /// <returns>True if the IDs match and GPU requirement is satisfied.</returns>
        public bool IdMatch(Guid id, bool requireOnGpu = false)
        {
            return this.Id.Equals(id) && (requireOnGpu ? this.OnGpu : true);
        }

    }
}
