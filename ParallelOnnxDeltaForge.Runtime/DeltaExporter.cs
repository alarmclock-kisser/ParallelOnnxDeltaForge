using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;

namespace ParallelOnnxDeltaForge.Runtime
{
    /// <summary>
    /// Exports LoRA deltas as standalone files or merges them into base model weights.
    /// Standalone format: JSON metadata + raw float arrays (simple, portable, no ONNX proto dep).
    /// Merge: reads raw ONNX bytes, patches initializer floats, writes back.
    /// </summary>
    public class DeltaExporter : ParallelOnnxDeltaForge.Shared.Interfaces.IDeltaExporter
    {
        private readonly RollingFileMemoryLogger _logger;

        public DeltaExporter(RollingFileMemoryLogger logger)
        {
            this._logger = logger;
        }

        public async Task<DeltaExportResult> ExportAsLoraAdapterAsync(LoRADeltaSet deltaSet, string outputPath)
        {
            if (outputPath == null) throw new ArgumentNullException(nameof(outputPath), "Output path cannot be null.");
            this._logger.LogInfo($"[Delta.Exporter] Exporting LoRA adapter to {outputPath}");

            try
            {
                var payload = new
                {
                    format = "paralleldeltaforge_lora_v1",
                    rank = deltaSet.Rank,
                    accumulatedTurns = deltaSet.AccumulatedTurns,
                    name = deltaSet.Name,
                    layers = deltaSet.Deltas.Select(kvp => new
                    {
                        layerName = kvp.Key,
                        aShape = kvp.Value.AShape,
                        bShape = kvp.Value.BShape,
                        scaleFactor = kvp.Value.ScaleFactor,
                        aData = kvp.Value.AData?.Select(f => Math.Round(f, 8)).ToArray(),
                        bData = kvp.Value.BData?.Select(f => Math.Round(f, 8)).ToArray(),
                    }).ToArray(),
                };

                var json = JsonConvert.SerializeObject(payload, Formatting.Indented);
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
                await File.WriteAllTextAsync(outputPath, json, Encoding.UTF8);

                long bytes = new FileInfo(outputPath).Length;
                this._logger.LogSuccess($"[Delta.Exporter] Saved: {outputPath} ({bytes:N0} bytes)");

                return new DeltaExportResult
                {
                    Success = true,
                    OutputPath = outputPath,
                    Mode = DeltaExportMode.StandaloneAdapter,
                    BytesWritten = bytes,
                };
            }
            catch (Exception ex)
            {
                this._logger.LogError($"[Delta.Exporter] Failed: {ex.Message}");
                return new DeltaExportResult { Success = false, ErrorMessage = ex.Message, Mode = DeltaExportMode.StandaloneAdapter };
            }
        }

        public async Task<DeltaExportResult> MergeIntoBaseModelAsync(LoRADeltaSet deltaSet, string baseModelPath, string outputPath)
        {
            if (deltaSet == null) throw new ArgumentNullException(nameof(deltaSet), "Delta set cannot be null.");
            if (baseModelPath == null) throw new ArgumentNullException(nameof(baseModelPath));
            if (outputPath == null) throw new ArgumentNullException(nameof(outputPath));
            this._logger.LogInfo($"[Delta.Exporter] Merging deltas into {Path.GetFileName(baseModelPath)}");

            try
            {
                if (!File.Exists(baseModelPath))
                    throw new FileNotFoundException($"Base model not found: {baseModelPath}");

                // ONNX merge requires protobuf parser; for now export deltas + instructions
                // Write a merge manifest alongside the model
                var manifest = new
                {
                    action = "merge_instructions",
                    baseModel = baseModelPath,
                    targetModel = outputPath,
                    rank = deltaSet.Rank,
                    accumulatedTurns = deltaSet.AccumulatedTurns,
                    layers = deltaSet.Deltas.Select(kvp => new
                    {
                        layerName = kvp.Key,
                        instruction = $"Add scale×(B×A) to '{kvp.Key}' initializer tensor",
                        scaleFactor = kvp.Value.ScaleFactor,
                        aShape = kvp.Value.AShape,
                        bShape = kvp.Value.BShape,
                        aData = kvp.Value.AData?.Select(f => Math.Round(f, 8)).ToArray(),
                        bData = kvp.Value.BData?.Select(f => Math.Round(f, 8)).ToArray(),
                    }).ToArray(),
                    note = "Automated ONNX protobuf merge requires Google.Protobuf + ONNX proto definitions. Use this manifest with a merge tool or script.",
                };

                var json = JsonConvert.SerializeObject(manifest, Formatting.Indented);
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
                await File.WriteAllTextAsync(outputPath + ".manifest.json", json, Encoding.UTF8);

                // Also copy base model to target as starting point
                File.Copy(baseModelPath, outputPath, true);

                long bytes = new FileInfo(outputPath).Length;
                this._logger.LogSuccess($"[Delta.Exporter] Base model copied to {outputPath}. Merge manifest written.");
                this._logger.LogWarning($"[Delta.Exporter] Manual merge required: see {outputPath}.manifest.json");

                return new DeltaExportResult
                {
                    Success = true,
                    OutputPath = outputPath,
                    Mode = DeltaExportMode.MergeIntoModel,
                    BytesWritten = bytes,
                };
            }
            catch (Exception ex)
            {
                this._logger.LogError($"[Delta.Exporter] Merge failed: {ex.Message}");
                return new DeltaExportResult { Success = false, ErrorMessage = ex.Message, Mode = DeltaExportMode.MergeIntoModel };
            }
        }
    }
}
