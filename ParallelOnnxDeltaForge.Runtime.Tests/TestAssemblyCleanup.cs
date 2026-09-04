using Microsoft.VisualStudio.TestTools.UnitTesting;
using ParallelOnnxDeltaForge.Shared.Testing;

namespace ParallelOnnxDeltaForge.Runtime.Tests;

[TestClass]
public static class TestAssemblyCleanup
{
    [AssemblyCleanup]
    public static void Flush()
    {
        TestResultCollector.FlushAll();
    }
}
