using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Buffers.Binary;
using NAudio.Wave;
using NAudio.CoreAudioApi;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.MediaDtos;
using System.Runtime.Versioning;
using ParallelOnnxDeltaForge.Shared.Interfaces;

namespace ParallelOnnxDeltaForge.Media
{
    public class AudioCollection : IMediaCollection, IDisposable, IAsyncDisposable
    {
        /// <summary>
        /// Gets or sets the export directory for this media collection instance.
        /// </summary>
        public string ExportDirectory { get; set; } = Path.GetFullPath(Environment.GetEnvironmentVariable("SHARPAI_AUDIO_EXPORT_DIR") ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyMusic), "SharpAI_AudioExports"));


        private readonly ConcurrentDictionary<Guid, AudioObj> _audios = [];
        public IReadOnlyCollection<AudioObj> Audios => this._audios.Values.ToList();

        /// <summary>
        /// Explicit implementation of <see cref="IMediaCollection.Objects"/>.
        /// Returns the collection of audio objects as <see cref="IMediaObj"/>.
        /// </summary>
        public IReadOnlyCollection<IMediaObj> Objects => this._audios.Values.Select(a => (IMediaObj)a).ToList();

        private CancellationTokenSource? recordingCts;


        /// <summary>
        /// Explicit implementation of <see cref="IMediaCollection.this[Guid]"/>.
        /// Returns the audio object by Guid as <see cref="IMediaObj"/>.
        /// </summary>
        public IMediaObj? this[Guid id] => this._audios.TryGetValue(id, out AudioObj? audioObj) ? (IMediaObj?)audioObj : null;

        /// <summary>
        /// Explicit implementation of <see cref="IMediaCollection.this[int]"/>.
        /// Returns the audio object by index as <see cref="IMediaObj"/>.
        /// </summary>
        public IMediaObj? this[int index] => (index >= 0 && index < this._audios.Count) ? (IMediaObj?)this._audios.Values.ElementAt(index) : null;
        public AudioObj? this[string name, bool fuzzyMatch = true] => fuzzyMatch ? this._audios.Values.FirstOrDefault(a => a.Name.Contains(name, StringComparison.OrdinalIgnoreCase)) : this._audios.Values.FirstOrDefault(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase));

        public bool IsRecording => this.recordingCts != null && this.recordingCts?.IsCancellationRequested == false;

        public AudioCollection(string? customExportDir = null, string[]? additionalRessourcePaths = null)
        {
            if (!string.IsNullOrEmpty(customExportDir))
            {
                ExportDirectory = Path.GetFullPath(customExportDir);
            }
            if (additionalRessourcePaths != null)
            {
                foreach (var path in additionalRessourcePaths)
                {
                    // Get every file in the directory or if it's a file, just take that
                    if (Directory.Exists(path))
                    {
                        var files = Directory.GetFiles(path);
                        foreach (var file in files)
                        {
                            this.ImportAudio(file);
                        }
                    }
                    else if (File.Exists(path))
                    {
                        this.ImportAudio(path);
                    }
                }
            }
        }


        // Add & Import
        public bool AddAudio(AudioObj audioObj)
        {
            return this._audios.TryAdd(audioObj.Id, audioObj);
        }

        public AudioObj? ImportAudio(string filePath)
        {
            if (!File.Exists(filePath))
            {
                return null;
            }

            if (!Path.GetExtension(filePath).Equals(".wav", StringComparison.OrdinalIgnoreCase) &&
                !Path.GetExtension(filePath).Equals(".mp3", StringComparison.OrdinalIgnoreCase) &&
                !Path.GetExtension(filePath).Equals(".flac", StringComparison.OrdinalIgnoreCase) &&
                !Path.GetExtension(filePath).Equals(".ogg", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            AudioObj? audioObj = null;
            try
            {
                audioObj = new AudioObj(filePath);
                this.AddAudio(audioObj);
            }
            catch
            {
                audioObj = null;
            }

            return audioObj;
        }

        public async Task<AudioObj?> ImportAudioAsync(string filePath)
        {
            return await Task.Run(() =>
            {
                try
                {
                    var audioObj = new AudioObj(filePath);
                    this.AddAudio(audioObj);
                    return audioObj;
                }
                catch
                {
                    return null;
                }
            });
        }

        public AudioObj? CreateFromInfo(AudioInfo info, bool tryAdd = true, bool disposeIfFailedToAdd = true, bool emptyData = false, long? pointer = 0)
        {
            long length = long.TryParse(info.Length, out var len) ? len : 0;
            long ptr = pointer ?? (long.TryParse(info.Pointer, out var p) ? p : 0);
            AudioObj obj = new()
            {
                BitDepth = info.BitDepth,
                Channels = info.Channels,
                ChunkSize = info.ChunkSize,
                Length = length,
                Data = emptyData ? [] : new float[length],
                FilePath = info.FilePath,
                Name = info.Name,
                Overlap = info.Overlap,
                SampleRate = info.SampleRate,
                Pointer = ptr
            };

            if (tryAdd)
            {
                if (this._audios.TryAdd(obj.Id, obj))
                {
                    return obj;
                }
                else if (disposeIfFailedToAdd)
                {
                    obj.Dispose();
                    return null;
                }
            }

            return obj;
        }


        [SupportedOSPlatform("windows")]
        private MMDevice[] GetCaptureDevices()
        {
            using var enumerator = new MMDeviceEnumerator();
            return enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active).ToArray();
        }

        [SupportedOSPlatform("windows")]
        public int FindActiveMicrophoneIndex()
        {
            var devices = this.GetCaptureDevices();
            if (devices.Length == 0)
            {
                return -1;
            }

            int bestDevice = 0;
            float maxPeak = 0f;

            // Wir testen jedes Ger�t kurz (200ms)
            for (int i = 0; i < devices.Length; i++)
            {
                float currentPeak = this.TestDevicePeak(devices[i]);
                if (currentPeak > maxPeak)
                {
                    maxPeak = currentPeak;
                    bestDevice = i;
                }
            }

            return bestDevice;
        }

        [SupportedOSPlatform("windows")]
        private float TestDevicePeak(MMDevice device)
        {
            float peak = 0f;
            var captureFormat = new WaveFormat(44100, 16, 1);
            using var capture = new WasapiRecorderBuilder()
                .WithDevice(device)
                .WithSharedMode()
                .WithEventSync()
                .WithBufferLength(200)
                .WithFormat(captureFormat)
                .Build();

            // Zero-copy span over the WASAPI buffer; only valid for the duration of this callback.
            capture.DataAvailable += (ReadOnlySpan<byte> buffer, AudioClientBufferFlags flags, long devicePosition, long qpcPosition) =>
            {
                if (buffer.IsEmpty)
                {
                    return;
                }

                // 16-bit PCM: 2 bytes per sample
                for (int i = 0; i + 1 < buffer.Length; i += 2)
                {
                    float sampleFloat = Math.Abs(BinaryPrimitives.ReadInt16LittleEndian(buffer.Slice(i)) / 32768f);
                    if (sampleFloat > peak)
                    {
                        peak = sampleFloat;
                    }
                }
            };

            capture.StartRecording();
            Thread.Sleep(200); // Kurze Zeit lauschen
            capture.StopRecording();

            return peak;
        }

        [SupportedOSPlatform("windows")]
        public async Task<AudioObj?> RecordAudioAsync(int? deviceIndex = null, int sampleRate = 44100, int bitDepth = 16, int channels = 2, Action<float>? onLevel = null)
        {
            var wf = new WaveFormat(sampleRate, bitDepth, channels);
            return await this.RecordAudioAsync(wf, deviceIndex, onLevel).ConfigureAwait(false);
        }


        [SupportedOSPlatform("windows")]
        public async Task<AudioObj?> RecordAudioAsync(WaveFormat waveFormat, int? deviceIndex = null, Action<float>? onLevel = null)
        {
            if (deviceIndex == null)
            {
                deviceIndex = this.FindActiveMicrophoneIndex();
                if (deviceIndex == -1)
                {
                    await RollingFileMemoryLogger.Instance.LogAsync("No recording devices found.");
                    return null;
                }
            }

            if (this.recordingCts != null)
            {
                await RollingFileMemoryLogger.Instance.LogAsync("Recording already in progress.").ConfigureAwait(false);
                return null;
            }

            var tcs = new TaskCompletionSource<AudioObj>(TaskCreationOptions.RunContinuationsAsynchronously);
            this.recordingCts = new CancellationTokenSource();
            var ct = this.recordingCts.Token;

            var sampleList = new List<float>();

            MMDevice captureDevice;
            using (var enumerator = new MMDeviceEnumerator())
            {
                var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active).ToArray();
                if (deviceIndex.Value >= 0 && deviceIndex.Value < devices.Length)
                {
                    captureDevice = devices[deviceIndex.Value];
                }
                else
                {
                    captureDevice = devices.FirstOrDefault() ?? throw new InvalidOperationException("No capture devices available.");
                }
            }

            // NAudio 3.x: build a modern WasapiRecorder. Request the caller's format; in shared mode with
            // AutoConvertPcm the engine converts to it. We decode as 16-bit PCM (2 bytes per sample).
            var waveIn = new WasapiRecorderBuilder()
                .WithDevice(captureDevice)
                .WithSharedMode()
                .WithEventSync()
                .WithBufferLength(200)
                .WithFormat(waveFormat)
                .Build();

            // Zero-copy span over the WASAPI buffer; only valid for the duration of this callback.
            waveIn.DataAvailable += (ReadOnlySpan<byte> buffer, AudioClientBufferFlags flags, long devicePosition, long qpcPosition) =>
            {
                if (buffer.IsEmpty)
                {
                    return;
                }

                // Convert 16-bit PCM to floats (-1.0 .. 1.0), keep interleaved channels
                float peak = 0f;
                for (int i = 0; i + 1 < buffer.Length; i += 2)
                {
                    float sample = BinaryPrimitives.ReadInt16LittleEndian(buffer.Slice(i)) / 32768f;
                    sampleList.Add(sample);
                    peak = Math.Max(peak, Math.Abs(sample));
                }

                try { onLevel?.Invoke(Math.Clamp(peak, 0f, 1f)); } catch { }
            };

            waveIn.RecordingStopped += (s, e) =>
            {
                string name = $"Recording_{DateTime.Now:yyyyMMdd_HHmmss}";
                var audioObj = new AudioObj(sampleList.ToArray(), waveIn.WaveFormat.SampleRate, waveIn.WaveFormat.Channels, waveIn.WaveFormat.BitsPerSample, name);
                tcs.TrySetResult(audioObj);
            };

            waveIn.StartRecording();
            await RollingFileMemoryLogger.Instance.LogAsync($"Recording started on device {deviceIndex.Value} with format {waveIn.WaveFormat.SampleRate}Hz, {waveIn.WaveFormat.BitsPerSample}bit, {waveIn.WaveFormat.Channels}ch").ConfigureAwait(false);

            // Wait until cancellation requested
            try
            {
                await Task.Delay(Timeout.Infinite, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                try
                {
                    waveIn.StopRecording();
                }
                catch { }
            }

            await RollingFileMemoryLogger.Instance.LogAsync("Recording stopped. Processing audio...").ConfigureAwait(false);
            var result = await tcs.Task.ConfigureAwait(false);

            waveIn.Dispose();
            captureDevice.Dispose();
            this.recordingCts.Dispose();
            this.recordingCts = null;

            return result;
        }

        public bool StopRecording()
        {
            if (this.recordingCts == null)
            {
                return false;
            }

            try
            {
                this.recordingCts.Cancel();
                return true;
            }
            catch
            {
                return false;
            }
        }



        // Get estimate duration from file without fully loading it
        public static TimeSpan? GetAudioDuration(string filePath)
        {
            try
            {
                using var reader = new AudioFileReader(filePath);
                return reader.TotalTime;
            }
            catch
            {
                return null;
            }
        }




        public bool RemoveAudio(AudioObj audioObj, bool disposeRemoved = true)
        {
            if (this._audios.TryRemove(audioObj.Id, out var removed))
            {
                if (disposeRemoved)
                {
                    removed.Dispose();
                }
                return true;
            }
            return false;
        }

        public bool RemoveAudio(Guid audioId, bool disposeRemoved = true)
        {
            var audioObj = this._audios[audioId];
            if (audioObj != null)
            {
                if (disposeRemoved)
                {
                    audioObj.Dispose();
                }
                return true;
            }
            return false;
        }

        public bool RemoveAudio(string name, bool fuzzyMatch = false, bool disposeRemoved = true)
        {
            AudioObj? audioObj = this[name, fuzzyMatch];
            if (audioObj != null)
            {
                if (disposeRemoved)
                {
                    audioObj.Dispose();
                }
                return true;
            }
            return false;
        }


        public int ClearAudios()
        {
            int count = this._audios.Count;
            foreach (var audio in this._audios.Values)
            {
                audio.Dispose();
            }
            this._audios.Clear();
            return count;
        }

        public async Task ClearAudiosAsync()
        {
            var disposeTasks = this._audios.Values.Select(a => Task.Run(() => a.Dispose())).ToArray();
            await Task.WhenAll(disposeTasks);
            this._audios.Clear();
        }

        public void Dispose()
        {
            foreach (var audio in this._audios.Values)
                audio.Dispose();
            this._audios.Clear();
            GC.SuppressFinalize(this);
        }

        public async ValueTask DisposeAsync()
        {
            var disposeTasks = this._audios.Values.Select(a => Task.Run(() => a.Dispose())).ToArray();

            await Task.WhenAll(disposeTasks);

            this._audios.Clear();

            GC.SuppressFinalize(this);
        }
    }
}
