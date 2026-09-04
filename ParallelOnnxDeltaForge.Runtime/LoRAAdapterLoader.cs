using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;

namespace ParallelOnnxDeltaForge.Runtime
{
    /// <summary>
    /// Loads LoRA adapter ONNX files and extracts per-layer A/B weight tensors.
    /// Convention: tensor names contain layer prefix + "lora_A" / "lora_B".
    /// </summary>
    public class LoRAAdapterLoader : ParallelOnnxDeltaForge.Shared.Interfaces.ILoRAAdapterLoader
    {
        private readonly RollingFileMemoryLogger _logger;
        private readonly Dictionary<string, LoraAdapterInfo> _loaded = new();

        public LoRAAdapterLoader(RollingFileMemoryLogger logger)
        {
            this._logger = logger;
        }

        public async Task<LoraAdapterInfo> LoadAsync(string adapterPath, string name, int rank, float scaleFactor)
        {
            if (!File.Exists(adapterPath))
                throw new FileNotFoundException($"LoRA adapter not found: {adapterPath}");

            this._logger.LogInfo($"[LoRA.Loader] Loading '{name}' from {Path.GetFileName(adapterPath)} (rank={rank}, scale={scaleFactor})");

            var info = new LoraAdapterInfo
            {
                AdapterPath = adapterPath,
                Name = name,
                Rank = rank,
                ScaleFactor = scaleFactor,
            };

            using var opts = new SessionOptions();
            opts.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;

            using var session = new InferenceSession(adapterPath, opts);

            var targetLayers = this.ExtractTargetLayers(session);
            info.TargetLayers = targetLayers;

            this._loaded[info.Id.ToString()] = info;
            this._logger.LogSuccess($"[LoRA.Loader] Loaded '{name}': {targetLayers.Count} target layers");

            return await Task.FromResult(info);
        }

        public IReadOnlyDictionary<string, LoraAdapterInfo> GetLoadedAdapters()
            => new Dictionary<string, LoraAdapterInfo>(this._loaded);

        private List<string> ExtractTargetLayers(InferenceSession session)
        {
            var layers = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var key in session.InputMetadata.Keys)
            {
                var layer = this.ExtractLayerFromTensor(key);
                if (layer != null) layers.Add(layer);
            }

            if (layers.Count == 0)
            {
                layers.Add("default");
                this._logger.LogWarning("[LoRA.Loader] No layers detected, using 'default'");
            }

            return layers.ToList();
        }

        private string? ExtractLayerFromTensor(string name)
        {
            var idx = name.IndexOf("lora", StringComparison.OrdinalIgnoreCase);
            if (idx > 0) return name.Substring(0, idx).TrimEnd('.', '_');

            idx = name.IndexOf("adapter", StringComparison.OrdinalIgnoreCase);
            if (idx > 0) return name.Substring(0, idx).TrimEnd('.', '_');

            return null;
        }
    }
}
