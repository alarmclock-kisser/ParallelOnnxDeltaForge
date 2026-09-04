using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using ParallelOnnxDeltaForge.Runtime;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;
using ParallelOnnxDeltaForge.Shared.Options;
using Shouldly;

namespace ParallelOnnxDeltaForge.Runtime.Tests;

[TestClass]
public class DeltaExporterTests : TestBase
{
    private DeltaExporter Create() => new(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));

    private LoRADeltaSet CreateDeltaSet(int rank = 4, int layerCount = 3)
    {
        var deltas = new System.Collections.Generic.Dictionary<string, LoRADelta>();
        for (int l = 0; l < layerCount; l++)
        {
            deltas[$"layer_{l}"] = new LoRADelta
            {
                LayerName = $"layer_{l}",
                AShape = new long[] { rank, 64 }, BShape = new long[] { 32, rank },
                AData = Enumerable.Range(0, rank * 64).Select(i => (float)(i % 10) / 10f).ToArray(),
                BData = Enumerable.Range(0, 32 * rank).Select(i => (float)(i % 7) / 7f).ToArray(),
                ScaleFactor = 1f,
            };
        }
        return new LoRADeltaSet { Name = "test_delta", Rank = rank, Deltas = deltas, AccumulatedTurns = 10 };
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_ValidDelta_ShouldWriteJsonFile()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_{Guid.NewGuid():N}.json");
        var result = await this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(), tmp);

        result.Success.ShouldBeTrue();
        result.OutputPath.ShouldBe(tmp);
        result.Mode.ShouldBe(DeltaExportMode.StandaloneAdapter);
        File.Exists(tmp).ShouldBeTrue();

        try
        {
            var json = File.ReadAllText(tmp);
            var el = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(json);
            el.GetProperty("format").GetString().ShouldBe("paralleldeltaforge_lora_v1");
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    [DataRow(8)]
    [DataRow(16)]
    public async Task ExportAsLoraAdapterAsync_VariousRanks_ShouldSucceed(int rank)
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_r{rank}_{Guid.NewGuid():N}.json");
        var result = await this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(rank), tmp);

        result.Success.ShouldBeTrue();
        try
        {
            var el = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(tmp));
            el.GetProperty("rank").GetInt32().ShouldBe(rank);
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_NullData_ShouldHandleGracefully()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_null_{Guid.NewGuid():N}.json");
        var deltas = new LoRADeltaSet
        {
            Name = "null_data", Rank = 4, AccumulatedTurns = 1,
            Deltas = new System.Collections.Generic.Dictionary<string, LoRADelta>
            { ["null_layer"] = new LoRADelta { LayerName = "null_layer" } }
        };
        var result = await this.Create().ExportAsLoraAdapterAsync(deltas, tmp);
        result.Success.ShouldBeTrue();
        if (File.Exists(tmp)) File.Delete(tmp);
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_NonExistentDirectory_ShouldCreateIt()
    {
        var dir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        var tmp = Path.Combine(dir, "adapter.json");
        var result = await this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(), tmp);

        result.Success.ShouldBeTrue();
        Directory.Exists(dir).ShouldBeTrue();
        try { File.Exists(tmp).ShouldBeTrue(); }
        finally { Directory.Delete(dir, true); }
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_BytesWritten_ShouldBePositive()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_bytes_{Guid.NewGuid():N}.json");
        var result = await this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(), tmp);
        result.BytesWritten.ShouldBeGreaterThan(0);
        if (File.Exists(tmp)) File.Delete(tmp);
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_EmptyDeltas_ShouldStillWrite()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_empty_{Guid.NewGuid():N}.json");
        var result = await this.Create().ExportAsLoraAdapterAsync(new LoRADeltaSet { Name = "empty", Rank = 4, AccumulatedTurns = 0 }, tmp);
        result.Success.ShouldBeTrue();
        File.Exists(tmp).ShouldBeTrue();
        if (File.Exists(tmp)) File.Delete(tmp);
    }

    [TestMethod]
    public async Task MergeIntoBaseModelAsync_BaseNotFound_ShouldReturnFailure()
    {
        var result = await this.Create().MergeIntoBaseModelAsync(this.CreateDeltaSet(), "C:\\nonexistent\\model.onnx", "output.onnx");
        result.Success.ShouldBeFalse();
        result.ErrorMessage.ShouldNotBeNullOrEmpty();
    }

    [TestMethod]
    public async Task MergeIntoBaseModelAsync_ValidBase_ShouldCopyAndWriteManifest()
    {
        var basePath = Path.Combine(Path.GetTempPath(), $"base_{Guid.NewGuid():N}.onnx");
        var outputPath = Path.Combine(Path.GetTempPath(), $"merged_{Guid.NewGuid():N}.onnx");
        File.WriteAllText(basePath, "FAKE_MODEL_DATA");
        var result = await this.Create().MergeIntoBaseModelAsync(this.CreateDeltaSet(), basePath, outputPath);

        result.Success.ShouldBeTrue();
        File.Exists(outputPath).ShouldBeTrue();
        File.Exists(outputPath + ".manifest.json").ShouldBeTrue();
        var el = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(outputPath + ".manifest.json"));
        el.GetProperty("action").GetString().ShouldBe("merge_instructions");
        File.Delete(outputPath);
        File.Delete(outputPath + ".manifest.json");
        if (File.Exists(basePath)) File.Delete(basePath);
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(3)]
    [DataRow(10)]
    public async Task MergeIntoBaseModelAsync_LayerCount_ShouldMatch(int layerCount)
    {
        var basePath = Path.Combine(Path.GetTempPath(), $"base_m_{Guid.NewGuid():N}.onnx");
        var outputPath = Path.Combine(Path.GetTempPath(), $"merged_m_{Guid.NewGuid():N}.onnx");
        File.WriteAllText(basePath, "FAKE");
        await this.Create().MergeIntoBaseModelAsync(this.CreateDeltaSet(layerCount: layerCount), basePath, outputPath);

        var el = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(outputPath + ".manifest.json"));
        el.GetProperty("layers").GetArrayLength().ShouldBe(layerCount);
        File.Delete(outputPath);
        File.Delete(outputPath + ".manifest.json");
        if (File.Exists(basePath)) File.Delete(basePath);
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_NullPath_ShouldFail()
    {
        await Should.ThrowAsync<Exception>(() => this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(), null!));
    }

    [TestMethod]
    public async Task MergeIntoBaseModelAsync_NullDeltas_ShouldThrow()
    {
        await Should.ThrowAsync<Exception>(() => this.Create().MergeIntoBaseModelAsync(null!, "x", "y"));
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_MetaData_ShouldBeAccurate()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"lora_meta_{Guid.NewGuid():N}.json");
        var deltas = this.CreateDeltaSet(12);
        deltas.AccumulatedTurns = 42;
        await this.Create().ExportAsLoraAdapterAsync(deltas, tmp);

        var el = System.Text.Json.JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(tmp));
        el.GetProperty("rank").GetInt32().ShouldBe(12);
        el.GetProperty("accumulatedTurns").GetInt32().ShouldBe(42);
        if (File.Exists(tmp)) File.Delete(tmp);
    }

    [TestMethod]
    public async Task ExportAsLoraAdapterAsync_ReadOnlyPath_ShouldNotCrash()
    {
        var dir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.SetAttributes(dir, FileAttributes.ReadOnly);
        var result = await this.Create().ExportAsLoraAdapterAsync(this.CreateDeltaSet(), Path.Combine(dir, "a.json"));
        result.ShouldNotBeNull();
        File.SetAttributes(dir, FileAttributes.Normal);
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
    }
}
