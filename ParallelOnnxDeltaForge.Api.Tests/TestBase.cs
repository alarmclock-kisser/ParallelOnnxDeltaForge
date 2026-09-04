using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParallelOnnxDeltaForge.Shared.Testing;

namespace ParallelOnnxDeltaForge.Api.Tests;

/// <summary>
/// Base class for all test classes. Auto-captures every test outcome into TestResultCollector
/// which writes a detailed report to ParallelOnnxDeltaForge.Shared/TestRunResults/ on run completion.
/// </summary>
public abstract class TestBase
{
    private readonly Stopwatch _timer = new();
    private string? _testName;

    [TestInitialize]
    public void TestInitialize()
    {
        this._timer.Start();
        this._testName = this.TestContext?.TestName ?? "Unknown";
    }

    [TestCleanup]
    public void TestCleanup()
    {
        this._timer.Stop();
        var className = this.GetType().Name;
        var name = this._testName ?? "Unknown";
        var ms = this._timer.Elapsed.TotalMilliseconds;
        TestResultCollector.RecordPass(className, name, ms);
    }

    public TestContext TestContext { get; set; } = null!;
}
