using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using ParallelOnnxDeltaForge.Api.Controllers;
using ParallelOnnxDeltaForge.Shared.Dtos;
using ParallelOnnxDeltaForge.Shared.Interfaces;
using Shouldly;

namespace ParallelOnnxDeltaForge.Api.Tests;

[TestClass]
public class DeltaForgeControllerTests : TestBase
{
    private DeltaForgeController CreateController(Mock<IOnnxDeltaForgeService> mock)
        => new(mock.Object);

    #region POST /load-model

    [TestMethod]
    public async Task LoadModelAsync_Success_ShouldReturnOkWithGuid()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var expectedGuid = Guid.NewGuid();
        mock.Setup(f => f.LoadModelAsync("C:\\m.onnx", 0)).ReturnsAsync(expectedGuid);
        var ctrl = this.CreateController(mock);

        var req = new LoadModelRequest { ModelPath = "C:\\m.onnx", DeviceId = 0 };

        // Act
        var result = await ctrl.LoadModelAsync(req);

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        ok.Value.ShouldBe(expectedGuid);
    }

    [TestMethod]
    public async Task LoadModelAsync_FileNotFound_ShouldReturn500()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.LoadModelAsync(It.IsAny<string>(), It.IsAny<int>()))
            .ThrowsAsync(new FileNotFoundException("Not found"));
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.LoadModelAsync(new LoadModelRequest { ModelPath = "x", DeviceId = 0 });

        // Assert
        var code = result.Result.ShouldBeOfType<ObjectResult>();
        code.StatusCode.ShouldBe(500);
        code.Value.ShouldBeOfType<ProblemDetails>();
    }

    [TestMethod]
    public async Task LoadModelAsync_EmptyModelPath_ShouldReturn500()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.LoadModelAsync(It.IsAny<string>(), It.IsAny<int>()))
            .ThrowsAsync(new ArgumentException("Empty path"));
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.LoadModelAsync(new LoadModelRequest { ModelPath = "", DeviceId = 0 });

        // Assert
        var code = result.Result.ShouldBeOfType<ObjectResult>();
        code.StatusCode.ShouldBe(500);
    }

    #endregion

    #region POST /load-model-with-lora

    [TestMethod]
    public async Task LoadModelWithLoraAsync_Success_ShouldReturnOk()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var expectedGuid = Guid.NewGuid();
        mock.Setup(f => f.LoadModelWithLoraAsync("m.onnx", "l.onnx", "name", 8, 1f, 0))
            .ReturnsAsync(expectedGuid);
        var ctrl = this.CreateController(mock);

        var req = new LoadModelLoraRequest
        {
            ModelPath = "m.onnx", LoraPath = "l.onnx", LoraName = "name",
            Rank = 8, ScaleFactor = 1f, DeviceId = 0
        };

        // Act
        var result = await ctrl.LoadModelWithLoraAsync(req);

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        ok.Value.ShouldBe(expectedGuid);
    }

    [TestMethod]
    [DataRow(-1)]
    [DataRow(99)]
    public async Task LoadModelWithLoraAsync_InvalidDeviceId_ShouldReturn500(int deviceId)
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.LoadModelWithLoraAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<float>(), It.IsAny<int>()))
            .ThrowsAsync(new ArgumentOutOfRangeException(nameof(deviceId)));
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.LoadModelWithLoraAsync(new LoadModelLoraRequest
        {
            ModelPath = "m.onnx", LoraPath = "l.onnx", DeviceId = deviceId
        });

        // Assert
        var code = result.Result.ShouldBeOfType<ObjectResult>();
        code.StatusCode.ShouldBe(500);
    }

    #endregion

    #region POST /load-lora-adapter

    [TestMethod]
    public async Task LoadLoraAdapterAsync_Success_ShouldReturnLoraAdapterInfo()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var expectedInfo = new LoraAdapterInfo { Name = "TestAdapter", Rank = 16, ScaleFactor = 2f };
        mock.Setup(f => f.LoadLoraAdapterAsync("adapter.onnx", "TestAdapter", 16, 2f))
            .ReturnsAsync(expectedInfo);
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.LoadLoraAdapterAsync(new LoadLoraRequest
        { AdapterPath = "adapter.onnx", Name = "TestAdapter", Rank = 16, ScaleFactor = 2f });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var info = ok.Value.ShouldBeOfType<LoraAdapterInfo>();
        info.Name.ShouldBe("TestAdapter");
        info.Rank.ShouldBe(16);
        info.ScaleFactor.ShouldBe(2f);
    }

    [TestMethod]
    [DataRow(0)]
    [DataRow(-1)]
    [DataRow(1000)]
    public async Task LoadLoraAdapterAsync_VariousRanks_ShouldPassThrough(int rank)
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.LoadLoraAdapterAsync(It.IsAny<string>(), It.IsAny<string>(), rank, It.IsAny<float>()))
            .ThrowsAsync(new ArgumentException($"Invalid rank: {rank}"));
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.LoadLoraAdapterAsync(new LoadLoraRequest { Rank = rank });

        // Assert
        var code = result.Result.ShouldBeOfType<ObjectResult>();
        code.StatusCode.ShouldBe(500);
    }

    #endregion

    #region POST /infer

    [TestMethod]
    public async Task RunInferenceAsync_Tracked_ShouldReturnResponse()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var expected = new InferenceResponse { Output = "hello", WasTracked = true, TurnIndex = 5 };
        mock.Setup(f => f.RunInferenceAsync(It.IsAny<InferenceRequest>())).ReturnsAsync(expected);
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.RunInferenceAsync(new InferenceRequest { Input = "hi", TrackForDelta = true });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var resp = ok.Value.ShouldBeOfType<InferenceResponse>();
        resp.Output.ShouldBe("hello");
        resp.WasTracked.ShouldBeTrue();
        resp.TurnIndex.ShouldBe(5);
    }

    [TestMethod]
    [DataRow(true)]
    [DataRow(false)]
    public async Task RunInferenceAsync_TrackSetting_ShouldPropagate(bool trackForDelta)
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.RunInferenceAsync(It.Is<InferenceRequest>(r => r.TrackForDelta == trackForDelta)))
            .ReturnsAsync(new InferenceResponse { WasTracked = trackForDelta });
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.RunInferenceAsync(new InferenceRequest { Input = "x", TrackForDelta = trackForDelta });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var resp = ok.Value.ShouldBeOfType<InferenceResponse>();
        resp.WasTracked.ShouldBe(trackForDelta);
    }

    [TestMethod]
    public async Task RunInferenceAsync_WithRawData_ShouldPassThrough()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.RunInferenceAsync(It.IsAny<InferenceRequest>()))
            .ReturnsAsync(new InferenceResponse { Output = "ok" });
        var ctrl = this.CreateController(mock);

        // Act
        await ctrl.RunInferenceAsync(new InferenceRequest
        {
            Input = "raw",
            InputData = new float[] { 1f, 2f, 3f },
            TrackForDelta = true
        });

        // Assert
        mock.Verify(f => f.RunInferenceAsync(It.Is<InferenceRequest>(
            r => r.InputData != null && r.InputData.Length == 3)), Times.Once);
    }

    #endregion

    #region POST /compute-deltas

    [TestMethod]
    public async Task ComputeDeltasAsync_ValidRank_ShouldReturnDeltaSet()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var expected = new LoRADeltaSet { Rank = 16, AccumulatedTurns = 10 };
        mock.Setup(f => f.ComputeDeltasAsync(16)).ReturnsAsync(expected);
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ComputeDeltasAsync(new ComputeDeltasRequest { TargetRank = 16 });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var deltaSet = ok.Value.ShouldBeOfType<LoRADeltaSet>();
        deltaSet.Rank.ShouldBe(16);
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    [DataRow(8)]
    [DataRow(32)]
    public async Task ComputeDeltasAsync_RankVariations_ShouldPassThrough(int rank)
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.ComputeDeltasAsync(rank))
            .ReturnsAsync(new LoRADeltaSet { Rank = rank });
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ComputeDeltasAsync(new ComputeDeltasRequest { TargetRank = rank });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var ds = ok.Value.ShouldBeOfType<LoRADeltaSet>();
        ds.Rank.ShouldBe(rank);
    }

    [TestMethod]
    public async Task ComputeDeltasAsync_NoContext_ShouldReturn500()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.ComputeDeltasAsync(It.IsAny<int>()))
            .ThrowsAsync(new ArgumentException("No context"));
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ComputeDeltasAsync(new ComputeDeltasRequest { TargetRank = 8 });

        // Assert
        var code = result.Result.ShouldBeOfType<ObjectResult>();
        code.StatusCode.ShouldBe(500);
    }

    #endregion

    #region POST /export-deltas

    [TestMethod]
    public async Task ExportDeltasAsync_Standalone_ShouldReturnSuccess()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var deltaSet = new LoRADeltaSet { Rank = 8 };
        var exportResult = new DeltaExportResult
        {
            Success = true, OutputPath = "x.json", Mode = DeltaExportMode.StandaloneAdapter, BytesWritten = 512
        };
        mock.Setup(f => f.ComputeDeltasAsync(8)).ReturnsAsync(deltaSet);
        mock.Setup(f => f.ExportDeltasAsync(deltaSet, DeltaExportMode.StandaloneAdapter, "out.json"))
            .ReturnsAsync(exportResult);
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ExportDeltasAsync(new ExportDeltasRequest
        { TargetRank = 8, Mode = DeltaExportMode.StandaloneAdapter, OutputPath = "out.json" });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var exported = ok.Value.ShouldBeOfType<DeltaExportResult>();
        exported.Success.ShouldBeTrue();
        exported.Mode.ShouldBe(DeltaExportMode.StandaloneAdapter);
    }

    [TestMethod]
    public async Task ExportDeltasAsync_Merge_ShouldReturnSuccess()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var deltaSet = new LoRADeltaSet { Rank = 4 };
        mock.Setup(f => f.ComputeDeltasAsync(4)).ReturnsAsync(deltaSet);
        mock.Setup(f => f.ExportDeltasAsync(deltaSet, DeltaExportMode.MergeIntoModel, "merged.onnx"))
            .ReturnsAsync(new DeltaExportResult
            { Success = true, OutputPath = "merged.onnx", Mode = DeltaExportMode.MergeIntoModel });
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ExportDeltasAsync(new ExportDeltasRequest
        { TargetRank = 4, Mode = DeltaExportMode.MergeIntoModel, OutputPath = "merged.onnx" });

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var exported = ok.Value.ShouldBeOfType<DeltaExportResult>();
        exported.Mode.ShouldBe(DeltaExportMode.MergeIntoModel);
    }

    #endregion

    #region GET /context

    [TestMethod]
    public void GetContext_Empty_ShouldReturnEmptyList()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.GetContext()).Returns(new List<ContextTurn>().AsReadOnly());
        var ctrl = this.CreateController(mock);

        // Act
        var result = ctrl.GetContext();

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var list = ((OkObjectResult)ok).Value.ShouldBeAssignableTo<IReadOnlyList<ContextTurn>>();
        list.Count.ShouldBe(0);
    }

    [TestMethod]
    public void GetContext_WithTurns_ShouldReturnAll()
    {
        // Arrange
        var turns = new List<ContextTurn>
        {
            new() { TurnIndex = 0, Input = "a" },
            new() { TurnIndex = 1, Input = "b" },
        }.AsReadOnly();

        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.GetContext()).Returns(turns);
        var ctrl = this.CreateController(mock);

        // Act
        var result = ctrl.GetContext();

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        var list = ((OkObjectResult)ok).Value.ShouldBeAssignableTo<IReadOnlyList<ContextTurn>>();
        list.Count.ShouldBe(2);
    }

    #endregion

    #region GET /adapters

    [TestMethod]
    public void GetAdapters_Initial_ShouldReturnEmpty()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.GetLoadedAdapters()).Returns(new List<LoraAdapterInfo>().AsReadOnly());
        var ctrl = this.CreateController(mock);

        // Act
        var result = ctrl.GetAdapters();

        // Assert
        var ok = result.Result.ShouldBeOfType<OkObjectResult>();
        ((OkObjectResult)ok).Value.ShouldBeAssignableTo<IReadOnlyList<LoraAdapterInfo>>();
    }

    #endregion

    #region POST /clear-context

    [TestMethod]
    public async Task ClearContextAsync_ShouldReturn204()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        mock.Setup(f => f.ClearContextAsync()).Returns(Task.CompletedTask);
        var ctrl = this.CreateController(mock);

        // Act
        var result = await ctrl.ClearContextAsync();

        // Assert
        result.ShouldBeOfType<NoContentResult>();
    }

    #endregion

    #region DELETE /unload

    [TestMethod]
    public void UnloadModel_WithSessionId_ShouldReturn204()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var sid = Guid.NewGuid();
        var ctrl = this.CreateController(mock);

        // Act
        var result = ctrl.UnloadModel(sid);

        // Assert
        result.ShouldBeOfType<NoContentResult>();
        mock.Verify(f => f.UnloadModel(sid), Times.Once);
    }

    [TestMethod]
    public void UnloadModel_NullSessionId_ShouldPassNull()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var ctrl = this.CreateController(mock);

        // Act
        ctrl.UnloadModel(null);

        // Assert
        mock.Verify(f => f.UnloadModel(null), Times.Once);
    }

    [TestMethod]
    public void UnloadAll_ShouldReturn204()
    {
        // Arrange
        var mock = new Mock<IOnnxDeltaForgeService>();
        var ctrl = this.CreateController(mock);

        // Act
        var result = ctrl.UnloadAll();

        // Assert
        result.ShouldBeOfType<NoContentResult>();
        mock.Verify(f => f.UnloadAll(), Times.Once);
    }

    #endregion
}
