using System;
using System.Collections.Concurrent;
using System.Text;

namespace ParallelOnnxDeltaForge.Shared.Testing
{
    public static class TestResultCollector
    {
        private static readonly ConcurrentBag<PassEntry> _passes = new();
        private static readonly ConcurrentBag<FailEntry> _fails = new();
        private static readonly DateTime _start = DateTime.UtcNow;

        public static void RecordPass(string cls, string test, double? dur = null) =>
            _passes.Add(new PassEntry(cls, test, dur));

        public static void RecordFail(string cls, string test, Exception ex, double? dur = null) =>
            _fails.Add(new FailEntry(cls, test, ex, dur));

        public static void FlushAll()
        {
            var dir = ResolveDir();
            var ts = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var path = Path.Combine(dir, $"TestRun_{ts}.txt");
            var lines = BuildReport();
            File.WriteAllText(path, lines);
        }

        private static string ResolveDir()
        {
            var d = Path.Combine(AppContext.BaseDirectory, @"..\..\..\..\ParallelOnnxDeltaForge.Shared\TestRunResults");
            Directory.CreateDirectory(d);
            return d;
        }

        private static string BuildReport()
        {
            var sb = new StringBuilder();
            var passes = _passes.ToList();
            var fails = _fails.ToList();
            var now = DateTime.UtcNow;

            sb.AppendLine("================================================================================");
            sb.AppendLine("  TEST RUN REPORT  ParallelOnnxDeltaForge");
            sb.AppendLine($"  Start     : {_start:yyyy-MM-dd HH:mm:ss.fff} UTC");
            sb.AppendLine($"  End       : {now:yyyy-MM-dd HH:mm:ss.fff} UTC");
            sb.AppendLine($"  Elapsed   : {(now - _start).TotalSeconds:F2}s");
            sb.AppendLine($"  Passed    : {passes.Count}");
            sb.AppendLine($"  Failed    : {fails.Count}");
            sb.AppendLine("================================================================================");
            sb.AppendLine();

            // Passes
            for (int i = 0; i < passes.Count; i++)
                sb.AppendLine($"  [{i + 1:000}] PASS | {passes[i].ClassName}.{passes[i].TestName}");
            sb.AppendLine();

            if (fails.Count == 0)
            {
                sb.AppendLine("  ALL TESTS PASSED  OK");
                sb.AppendLine();
            }
            else
            {
                sb.AppendLine($"  {fails.Count} FAILED TEST(S) SEE DETAILS BELOW");
                sb.AppendLine("================================================================================");
                sb.AppendLine();

                foreach (var f in fails)
                    WriteFailDetail(sb, f);
            }

            sb.AppendLine("================================================================================");
            sb.AppendLine("  END OF REPORT");
            sb.AppendLine("================================================================================");

            return sb.ToString();
        }

        private static void WriteFailDetail(StringBuilder sb, FailEntry f)
        {
            sb.AppendLine($"  FAILURE  {f.ClassName}.{f.TestName}");
            sb.AppendLine($"  Timestamp   {DateTime.UtcNow:HH:mm:ss.fff}");
            if (f.DurationMs.HasValue)
                sb.AppendLine($"  Duration    {f.DurationMs.Value:F2}ms");
            sb.AppendLine($"  Exception   {f.Exception.GetType().FullName}");
            sb.AppendLine($"  Message     {f.Exception.Message}");
            sb.AppendLine();

            // Inner exceptions
            sb.AppendLine("  Inner Exceptions:");
            Exception? inner = f.Exception.InnerException;
            int depth = 0;
            while (inner != null && depth < 10)
            {
                sb.AppendLine($"    [{depth}] {inner.GetType().Name}: {inner.Message}");
                inner = inner.InnerException;
                depth++;
            }
            if (inner != null)
                sb.AppendLine("    ... depth limit");
            if (depth == 0)
                sb.AppendLine("    (none)");
            sb.AppendLine();

            // Stack trace
            sb.AppendLine("  Stack Trace:");
            if (f.Exception.StackTrace != null)
            {
                var parts = f.Exception.StackTrace.Split('\n');
                for (int i = 0; i < parts.Length; i++)
                    sb.AppendLine($"    {parts[i].Trim()}");
            }
            else
                sb.AppendLine("    (none)");

            sb.AppendLine();
            sb.AppendLine(new string('-', 80));
            sb.AppendLine();
        }

        private sealed record PassEntry(string ClassName, string TestName, double? DurationMs);
        private sealed record FailEntry(string ClassName, string TestName, Exception Exception, double? DurationMs);
    }
}
