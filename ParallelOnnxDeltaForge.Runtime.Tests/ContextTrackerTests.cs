using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParallelOnnxDeltaForge.Runtime;
using ParallelOnnxDeltaForge.Shared.Dtos;
using Shouldly;

namespace ParallelOnnxDeltaForge.Runtime.Tests;

[TestClass]
public class ContextTrackerTests : TestBase
{
    [TestMethod]
    public void Constructor_TurnCount_ShouldBeZero()
    {
        var tracker = new ContextTracker();
        tracker.TurnCount.ShouldBe(0);
    }

    [TestMethod]
    public void RecordTurn_SingleTurn_ShouldIncrementCount()
    {
        var tracker = new ContextTracker();
        tracker.RecordTurn(new ContextTurn { Input = "hello", Output = "world" });
        tracker.TurnCount.ShouldBe(1);
    }

    [TestMethod]
    public void RecordTurn_MultipleTurns_ShouldReturnAllOrdered()
    {
        var tracker = new ContextTracker();
        for (int i = 0; i < 10; i++)
            tracker.RecordTurn(new ContextTurn { Input = $"input_{i}" });

        var turns = tracker.GetTurns();
        turns.Count.ShouldBe(10);
        for (int i = 0; i < 10; i++)
            turns[i].TurnIndex.ShouldBe(i);
    }

    [TestMethod]
    public void RecordTurn_ShouldAssignTimestampAndIndex()
    {
        var tracker = new ContextTracker();
        var turn = new ContextTurn { Input = "test" };
        tracker.RecordTurn(turn);

        turn.TurnIndex.ShouldBe(0);
        turn.Timestamp.ShouldNotBe(default);
        (DateTime.UtcNow - turn.Timestamp).TotalSeconds.ShouldBeLessThan(2);
    }

    [TestMethod]
    public void Clear_ShouldResetEverything()
    {
        var tracker = new ContextTracker();
        tracker.RecordTurn(new ContextTurn { Input = "a" });
        tracker.RecordTurn(new ContextTurn { Input = "b" });
        tracker.Clear();

        tracker.TurnCount.ShouldBe(0);
        tracker.GetTurns().Count.ShouldBe(0);
    }

    [TestMethod]
    public void Clear_ThenRecord_ShouldStartIndexAtZero()
    {
        var tracker = new ContextTracker();
        tracker.RecordTurn(new ContextTurn { Input = "a" });
        tracker.Clear();
        tracker.RecordTurn(new ContextTurn { Input = "b" });

        var turns = tracker.GetTurns();
        turns.Count.ShouldBe(1);
        turns[0].TurnIndex.ShouldBe(0);
    }

    [TestMethod]
    public void GetTurns_Empty_ShouldReturnEmptyList()
    {
        var tracker = new ContextTracker();
        tracker.GetTurns().Count.ShouldBe(0);
    }

    [TestMethod]
    public void RecordTurn_WithData_ShouldPreserveData()
    {
        var tracker = new ContextTracker();
        var input = new float[] { 1f, 2f, 3f };
        var baseOut = new float[] { 0.1f, 0.2f };
        var loraOut = new float[] { 0.3f, 0.4f };
        var turn = new ContextTurn
        {
            Input = "test", InputData = input, BaseOutputData = baseOut, LoraOutputData = loraOut
        };

        tracker.RecordTurn(turn);
        var result = tracker.GetTurns().Single();
        result.InputData.ShouldBe(input);
        result.BaseOutputData.ShouldBe(baseOut);
        result.LoraOutputData.ShouldBe(loraOut);
    }

    [TestMethod]
    [DataRow(1000)]
    [DataRow(5000)]
    [DataRow(10000)]
    public void RecordTurn_LargeVolume_ShouldHandleConcurrency(int count)
    {
        var tracker = new ContextTracker();
        var tasks = Enumerable.Range(0, count)
            .Select(i => Task.Run(() => tracker.RecordTurn(new ContextTurn { Input = $"t_{i}" })))
            .ToArray();
        Task.WaitAll(tasks);

        tracker.TurnCount.ShouldBe(count);
    }

    [TestMethod]
    public void RecordTurn_NullData_ShouldNotThrow()
    {
        var tracker = new ContextTracker();
        Should.NotThrow(() => tracker.RecordTurn(new ContextTurn
        { Input = "test", InputData = null, BaseOutputData = null, LoraOutputData = null }));
    }

    [TestMethod]
    public void Clear_MultipleTimes_ShouldNotThrow()
    {
        var tracker = new ContextTracker();
        Should.NotThrow(() =>
        {
            tracker.Clear();
            tracker.Clear();
            tracker.Clear();
        });
    }

    [TestMethod]
    public void GetTurns_ReturnsReadOnly_ShouldNotAllowModification()
    {
        var tracker = new ContextTracker();
        tracker.RecordTurn(new ContextTurn { Input = "a" });
        var turns = tracker.GetTurns();

        Should.Throw<NotSupportedException>(() =>
        {
            ((IList<ContextTurn>)turns).Add(new ContextTurn { Input = "hack" });
        });
    }
}
