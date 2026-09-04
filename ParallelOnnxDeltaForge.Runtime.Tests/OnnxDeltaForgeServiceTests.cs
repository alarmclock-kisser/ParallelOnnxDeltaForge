using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using ParallelOnnxDeltaForge.Runtime;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;
using ParallelOnnxDeltaForge.Shared.Options;
using Shouldly;

namespace ParallelOnnxDeltaForge.Runtime.Tests;

[TestClass]
public class OnnxDeltaForgeServiceTests : TestBase
{
    private Mock<OnnxGpuService> MockGpu()
    {
        var m = new Mock<OnnxGpuService>(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        var sid = Guid.NewGuid();
        m.Setup(s => s.LoadModelAsync(It.IsAny<string>(), It.IsAny<int>())).ReturnsAsync(sid);
        return m;
    }

    private OnnxDeltaForgeService Create(Mock<OnnxGpuService> gpu)
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        return new OnnxDeltaForgeService(gpu.Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
    }

    private LoRAAdapterLoader CreateLoader() =>
        new LoRAAdapterLoader(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));

    [TestMethod]
    public async Task LoadModelAsync_ValidPath_ShouldReturnSessionId()
    {
        var mock = this.MockGpu();
        var svc = this.Create(mock);

        var id = await svc.LoadModelAsync("C:\\model.onnx", 0);
        id.ShouldNotBe(Guid.Empty);
        mock.Verify(s => s.LoadModelAsync("C:\\model.onnx", 0), Times.Once);
    }

    [TestMethod]
    public async Task LoadModelAsync_InvalidPath_ShouldPropagate()
    {
        var mock = new Mock<OnnxGpuService>(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        mock.Setup(s => s.LoadModelAsync(It.IsAny<string>(), It.IsAny<int>())).ThrowsAsync(new FileNotFoundException("Not found"));
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(mock.Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);

        await Should.ThrowAsync<FileNotFoundException>(() => svc.LoadModelAsync("x.onnx", 0));
    }

    [TestMethod]
    public async Task LoadLoraAdapterAsync_FileNotFound_ShouldThrow()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);

        await Should.ThrowAsync<FileNotFoundException>(() => svc.LoadLoraAdapterAsync("C:\\nonexistent\\adapter.onnx", "x", 4, 1f));
    }

    [TestMethod]
    public async Task RunInferenceAsync_WithTrack_ShouldRecordTurn()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        var resp = await svc.RunInferenceAsync(new InferenceRequest { Input = "Hello", TrackForDelta = true });
        resp.WasTracked.ShouldBeTrue();
        resp.TurnIndex.ShouldBe(0);
    }

    [TestMethod]
    public async Task RunInferenceAsync_WithoutTrack_ShouldNotRecord()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        (await svc.RunInferenceAsync(new InferenceRequest { Input = "Hello", TrackForDelta = false }))
            .WasTracked.ShouldBeFalse();
    }

    [TestMethod]
    public async Task RunInferenceAsync_MultipleTurns_ShouldIncrementIndex()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        var r1 = await svc.RunInferenceAsync(new InferenceRequest { Input = "a", TrackForDelta = true });
        var r2 = await svc.RunInferenceAsync(new InferenceRequest { Input = "b", TrackForDelta = true });
        r1.TurnIndex.ShouldBe(0);
        r2.TurnIndex.ShouldBe(1);
    }

    [TestMethod]
    public async Task RunInferenceAsync_WithDuration_ShouldBePositive()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        (await svc.RunInferenceAsync(new InferenceRequest { Input = "test" })).DurationMs.ShouldBeGreaterThanOrEqualTo(0);
    }

    [TestMethod]
    public async Task RunInferenceAsync_WithRawData_ShouldTrackData()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);
        await svc.RunInferenceAsync(new InferenceRequest { Input = "raw", InputData = new float[] { 1f, 2f, 3f }, TrackForDelta = true });

        svc.GetContext().Count.ShouldBe(1);
        svc.GetContext()[0].InputData.ShouldNotBeNull();
    }

    [TestMethod]
    public async Task ComputeDeltasAsync_NoContext_ShouldThrow()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await Should.ThrowAsync<ArgumentException>(() => svc.ComputeDeltasAsync(8));
    }

    [TestMethod]
    public async Task ComputeDeltasAsync_WithContext_ShouldReturnSet()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        for (int i = 0; i < 5; i++)
            await svc.RunInferenceAsync(new InferenceRequest
            { Input = $"t_{i}", InputData = Enumerable.Range(0, 16).Select(j => (float)j).ToArray(), TrackForDelta = true });

        (await svc.ComputeDeltasAsync(4)).Rank.ShouldBe(4);
    }

    [TestMethod]
    public async Task ExportDeltasAsync_Standalone_ShouldReturnSuccess()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"export_{Guid.NewGuid():N}.json");
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);

        for (int i = 0; i < 3; i++)
            await svc.RunInferenceAsync(new InferenceRequest
            { Input = $"t_{i}", InputData = Enumerable.Range(0, 16).Select(j => (float)j).ToArray(), TrackForDelta = true });

        var deltaSet = await svc.ComputeDeltasAsync(4);
        var result = await svc.ExportDeltasAsync(deltaSet, DeltaExportMode.StandaloneAdapter, tmp);
        result.Success.ShouldBeTrue();
        result.Mode.ShouldBe(DeltaExportMode.StandaloneAdapter);
        if (File.Exists(tmp)) File.Delete(tmp);
    }

    [TestMethod]
    public async Task ExportDeltasAsync_Merge_NoModelLoaded_ShouldThrow()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        await Should.ThrowAsync<InvalidOperationException>(() =>
            svc.ExportDeltasAsync(new LoRADeltaSet { Rank = 4 }, DeltaExportMode.MergeIntoModel, "out.onnx"));
    }

    [TestMethod]
    public void GetContext_Empty_ShouldReturnEmptyList()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        svc.GetContext().Count.ShouldBe(0);
    }

    [TestMethod]
    public async Task ClearContextAsync_ShouldClearAll()
    {
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false })),
            new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false }));
        await svc.LoadModelAsync("model.onnx", 0);
        await svc.RunInferenceAsync(new InferenceRequest { Input = "a", TrackForDelta = true });
        await svc.ClearContextAsync();
        svc.GetContext().Count.ShouldBe(0);
    }

    [TestMethod]
    public void GetLoadedAdapters_Initial_ShouldReturnEmpty()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, new LoRAAdapterLoader(log), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        svc.GetLoadedAdapters().Count.ShouldBe(0);
    }

    [TestMethod]
    public void UnloadModel_WithSessionId_ShouldCallGpuService()
    {
        var mock = this.MockGpu();
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(mock.Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        var sid = Guid.NewGuid();
        svc.UnloadModel(sid);
        mock.Verify(s => s.UnloadModel(sid), Times.Once);
    }

    [TestMethod]
    public void UnloadModel_NullSessionId_ShouldPassThrough()
    {
        var mock = this.MockGpu();
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(mock.Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        svc.UnloadModel(null);
        mock.Verify(s => s.UnloadModel(null), Times.Once);
    }

    [TestMethod]
    public void UnloadAll_ShouldCallGpuService()
    {
        var mock = this.MockGpu();
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(mock.Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        svc.UnloadAll();
        mock.Verify(s => s.UnloadAll(), Times.Once);
    }

    [TestMethod]
    public void Dispose_ShouldNotThrow()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        Should.NotThrow(() => svc.Dispose());
    }

    [TestMethod]
    public void Dispose_Twice_ShouldNotThrow()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        svc.Dispose();
        Should.NotThrow(() => svc.Dispose());
    }

    [TestMethod]
    public async Task RunInferenceAsync_NoModelLoaded_ShouldNotCrash()
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        (await svc.RunInferenceAsync(new InferenceRequest { Input = "test" })).ShouldNotBeNull();
    }

    [TestMethod]
    [DataRow("")]
    [DataRow("   ")]
    public async Task RunInferenceAsync_EmptyInput_ShouldStillWork(string input)
    {
        var log = new RollingFileMemoryLogger(new RollingFileMemoryLoggerOptions { Silent = true, CreateLogFile = false });
        var svc = new OnnxDeltaForgeService(this.MockGpu().Object, this.CreateLoader(), new ContextTracker(),
            new LoRADeltaComputationService(), new DeltaExporter(log), log);
        await svc.LoadModelAsync("model.onnx", 0);
        (await svc.RunInferenceAsync(new InferenceRequest { Input = input })).ShouldNotBeNull();
    }
}
