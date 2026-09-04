using System;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParallelOnnxDeltaForge.Runtime;
using ParallelOnnxDeltaForge.Shared.Dtos;
using Shouldly;

namespace ParallelOnnxDeltaForge.Runtime.Tests;

[TestClass]
public class LoRADeltaComputationServiceTests : TestBase
{
    private LoRADeltaComputationService Create() => new();

    [TestMethod]
    public async Task EmptyTurns_ShouldThrow()
    {
        var svc = this.Create();
        await Should.ThrowAsync<ArgumentException>(() => svc.ComputeFromContextAsync(Array.Empty<ContextTurn>(), 8));
    }

    [TestMethod]
    public async Task NullData_ShouldReturnEmptySet()
    {
        var svc = this.Create();
        var turns = new[] { new ContextTurn { Input = "test" } };
        var result = await svc.ComputeFromContextAsync(turns, 4);

        result.AccumulatedTurns.ShouldBe(0);
        result.Deltas.Count.ShouldBe(0);
        result.Rank.ShouldBe(4);
    }

    [TestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(8)]
    [DataRow(16)]
    [DataRow(64)]
    public async Task VariousRanks_ShouldRespectRank(int rank)
    {
        var svc = this.Create();
        var turns = this.CreateTurns(32, 5);
        var result = await svc.ComputeFromContextAsync(turns, rank);
        result.Rank.ShouldBe(rank);
        result.AccumulatedTurns.ShouldBe(5);
    }

    [TestMethod]
    public async Task SingleTurn_ShouldProduceDeltas()
    {
        var svc = this.Create();
        var turns = this.CreateTurns(16, 1);
        var result = await svc.ComputeFromContextAsync(turns, 4);

        result.AccumulatedTurns.ShouldBe(1);
        result.Deltas.ShouldNotBeEmpty();
        var delta = result.Deltas.Values.First();
        delta.AData.ShouldNotBeNull();
        delta.BData.ShouldNotBeNull();
    }

    [TestMethod]
    public async Task MultipleTurns_ShouldAccumulate()
    {
        var svc = this.Create();
        var turns = this.CreateTurns(64, 10);
        var result = await svc.ComputeFromContextAsync(turns, 8);

        result.AccumulatedTurns.ShouldBe(10);
        result.Deltas.ShouldNotBeEmpty();
        result.Deltas.Values.First().AData!.Length.ShouldBeGreaterThan(0);
    }

    [TestMethod]
    [DataRow(4, 8)]
    [DataRow(8, 16)]
    [DataRow(16, 32)]
    [DataRow(32, 64)]
    public async Task RankVsDim_Combinations(int rank, int dim)
    {
        var svc = this.Create();
        var turns = this.CreateTurns(dim, 3);
        var result = await svc.ComputeFromContextAsync(turns, rank);
        result.ShouldNotBeNull();
        result.Rank.ShouldBe(rank);
    }

    [TestMethod]
    public async Task DeterministicInput_ShouldYieldSameResult()
    {
        var svc = this.Create();
        var turns = this.CreateDeterministicTurns(16, 3);
        var r1 = await svc.ComputeFromContextAsync(turns, 4);
        var r2 = await svc.ComputeFromContextAsync(turns, 4);

        var d1 = r1.Deltas.Values.First().AData!;
        var d2 = r2.Deltas.Values.First().AData!;
        d1.Length.ShouldBe(d2.Length);
        for (int i = 0; i < d1.Length; i++)
            d1[i].ShouldBe(d2[i]);
    }

    [TestMethod]
    public async Task VerySmallData_ShouldNotThrow()
    {
        var svc = this.Create();
        var turns = new[] { new ContextTurn
        {
            Input = "x",
            InputData = new float[] { 1f },
            BaseOutputData = new float[] { 0f },
            LoraOutputData = new float[] { 0.5f }
        }};
        var result = await svc.ComputeFromContextAsync(turns, 1);
        result.ShouldNotBeNull();
    }

    [TestMethod]
    public async Task AllZeros_ShouldNotThrow()
    {
        var svc = this.Create();
        var d = 32;
        var turns = new[] { new ContextTurn
        {
            Input = "x",
            InputData = new float[d],
            BaseOutputData = new float[d],
            LoraOutputData = new float[d]
        }};
        (await svc.ComputeFromContextAsync(turns, 4)).ShouldNotBeNull();
    }

    [TestMethod]
    public async Task MismatchedLengths_ShouldHandleGracefully()
    {
        var svc = this.Create();
        var turns = new[] { new ContextTurn
        {
            Input = "x",
            InputData = new float[] { 1f, 2f },
            BaseOutputData = new float[] { 0f, 0f, 0f },
            LoraOutputData = new float[] { 0.1f, 0.2f, 0.3f }
        }};
        (await svc.ComputeFromContextAsync(turns, 2)).ShouldNotBeNull();
    }

    [TestMethod]
    public async Task HugeDimension_ShouldComplete()
    {
        var svc = this.Create();
        var turns = this.CreateTurns(4096, 5);
        var result = await svc.ComputeFromContextAsync(turns, 16);
        result.AccumulatedTurns.ShouldBe(5);
    }

    [TestMethod]
    public async Task NegativeRank_ShouldNotExplode()
    {
        var svc = this.Create();
        var turns = this.CreateTurns(16, 3);
        await Should.NotThrowAsync(() => svc.ComputeFromContextAsync(turns, -1));
    }

    private ContextTurn[] CreateTurns(int dim, int count)
    {
        var rng = new Random(42);
        var turns = new ContextTurn[count];
        for (int t = 0; t < count; t++)
        {
            var input = new float[dim];
            var baseOut = new float[dim];
            var loraOut = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                input[i] = (float)rng.NextDouble();
                baseOut[i] = (float)rng.NextDouble();
                loraOut[i] = baseOut[i] + (float)rng.NextDouble() * 0.1f;
            }
            turns[t] = new ContextTurn
            {
                Input = $"turn_{t}", InputData = input,
                BaseOutputData = baseOut, LoraOutputData = loraOut
            };
        }
        return turns;
    }

    private ContextTurn[] CreateDeterministicTurns(int dim, int count)
    {
        var turns = new ContextTurn[count];
        for (int t = 0; t < count; t++)
        {
            var input = new float[dim];
            var baseOut = new float[dim];
            var loraOut = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                input[i] = (float)(i + 1) / dim;
                baseOut[i] = (float)i / dim;
                loraOut[i] = (float)(i + 1) / (dim + 1);
            }
            turns[t] = new ContextTurn
            {
                Input = $"det_{t}", InputData = input,
                BaseOutputData = baseOut, LoraOutputData = loraOut
            };
        }
        return turns;
    }
}
