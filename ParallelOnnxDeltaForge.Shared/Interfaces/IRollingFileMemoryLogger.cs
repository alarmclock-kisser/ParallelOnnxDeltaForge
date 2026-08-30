using ParallelOnnxDeltaForge.Shared.Options;

namespace ParallelOnnxDeltaForge.Shared.Interfaces
{
    public interface IRollingFileMemoryLogger
    {
        public string? LogFilePath { get; }
        public Action? SaveToRepositoryOnShutdown { get; }
        public RollingFileMemoryLoggerOptions Settings { get; }

        public event Action<DateTime, string>? LogWritten;

        public void AddComment(DateTime? capturedAt = null, TimeSpan? elapsedSince = null, string comment = "<!!!>");
        public Task AddCommentAsync(DateTime? capturedAt = null, TimeSpan? elapsedSince = null, string comment = "<!!!>", bool? configureAwait = null);
        public void ClearLogs();
        public string ConfigureSaveToRepository(bool? configureToggle = null, string? subDirOrDifferentPath = null, int maxPreviousLogFiles = 8, Action? onShutdown = null);
        public string[] GetAllLogFilePaths();
        public string GetInnerExceptionsRecursively(Exception ex);
        public string GetInnerExceptionsRecursively(Exception ex, int? maxDepth = null, bool appendStackTrace = true, string openingBracket = "(", string closingBracket = ")", string separator = " ", bool asSingleLine = false);
        public IReadOnlyList<string> GetLogLines(bool? returnFilteredLog = false, bool reverseOrder = false);
        public string[]? GetNamespacesForProject(string? projectName = null, bool includeSubNamespaces = true, bool ignoreCase = true);
        public string? GetPreviousLogFilePath(int backIndex = 0);
        public SynchronizationContext? GetUiContext(bool copy = false);
        public void InitializeLogger(RollingFileMemoryLoggerOptions? options = null, Action? onShutdown = null, CancellationToken? exitCancellationToken = null, SynchronizationContext? synchronizationContext = null, bool setGlobally = false);
        public void Log(Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true, string? preText = null);
        public void Log(string message);
        public void Log(string message, Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true);
        public Task LogAsync(Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true, string? preText = null, bool? configureAwait = null);
        public Task LogAsync(string message, bool? configureAwait = null);
        public void LogError(string message);
        public void LogInfo(string message);
        public void LogSuccess(string message);
        public void LogWarning(string message);
        public string ResolveRepositoryDirectory(string? projectName = null, string subPath = "Logs", bool ensureDirectoryExists = false);
        public string ResolveRepositoryLogFilePath(string? differentFilePathOrDirectory = null);
        public string SaveToRepository(string? differentFilePathOrDirectory = null, bool? returnFilteredLog = null, bool reverseOrder = false, bool forceSave = false);
        public void SetOnShutdownAction(Action? onShutdown, CancellationToken cancellationToken = default, bool setGlobally = false);
        public void SetUiContext(SynchronizationContext? context);
        public void StartBackgroundWriter(CancellationToken cancellationToken);
    }
}