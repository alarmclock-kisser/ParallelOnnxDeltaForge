using ParallelOnnxDeltaForge.Shared.Interfaces;
using ParallelOnnxDeltaForge.Shared.Options;
using ParallelOnnxDeltaForge.Shared.Utils;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.ComponentModel;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ConstrainedExecution;
using System.Runtime.Serialization;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace ParallelOnnxDeltaForge.Shared
{
    /// <summary>
    /// A logger class that provides thread-safe logging functionality for applications. It allows logging messages, exceptions, and user comments with timestamps, and supports filtering log entries based on a specified phrase. The logger can write log entries to a file, echo them to the console, and maintain a binding list for UI components. It also provides methods for saving logs to a repository and managing log files.
    /// </summary>
    public partial class RollingFileMemoryLogger : IRollingFileMemoryLogger
    {
        /// <summary>
        /// Global settings for the StaticLogger.
        /// </summary>
        public RollingFileMemoryLoggerOptions Settings { get; private set; } = new();


        /// <summary>
        /// Gets the singleton instance of the <see cref="RollingFileMemoryLogger"/> class. This instance can be used to log messages, exceptions, and comments throughout the application.
        /// </summary>
        public static readonly RollingFileMemoryLogger Instance = new();

        /// <summary>
        /// Indicates whether the instance settings should be applied globally. If true, the settings of the singleton instance will be used throughout the application; if false, each instance can have its own settings.
        /// </summary>
        public static bool ApplyInstanceSettingsGlobally { get; private set; } = true;

        /// <summary>
        /// Indicates whether the instance's shutdown action should be applied globally. If true, the shutdown action of the singleton instance will be invoked on application shutdown; if false, each instance can have its own shutdown action.
        /// </summary>
        public static bool ApplyInstanceOnShutdownGlobally { get; private set; } = true;



        /// <summary>
        /// Initializes a new instance of the <see cref="RollingFileMemoryLogger"/> class. If <paramref name="applyGlobalSettings"/> is true, the instance will use the global settings and shutdown action from the singleton instance; otherwise, it will use its own settings and shutdown action.
        /// </summary>
        /// <param name="settings">The settings to use for this instance. If null, the default settings will be used.</param>
        /// <param name="applyGlobalSettings">If true, the instance will use the global settings and shutdown action from the singleton instance; otherwise, it will use its own settings and shutdown action.</param>
        public RollingFileMemoryLogger(RollingFileMemoryLoggerOptions? settings = null, bool? applyGlobalSettings = null, Action? onShutdown = null, CancellationToken? exitCancellationToken = null, SynchronizationContext? synchronizationContext = null, bool setGlobally = false)
        {
            this.Settings = settings ?? new RollingFileMemoryLoggerOptions();

            if (applyGlobalSettings == true)
            {
                if (ApplyInstanceSettingsGlobally)
                {
                    this.Settings = Instance.Settings;
                }
                if (ApplyInstanceOnShutdownGlobally)
                {
                    this.SetOnShutdownAction(Instance.SaveToRepositoryOnShutdown, Instance._logCts is not null ? Instance._logCts.Token : CancellationToken.None, false);
                }
            }

            this._logChannel = System.Threading.Channels.Channel.CreateBounded<string>(new System.Threading.Channels.BoundedChannelOptions(this.Settings.MaxLogEntries ?? 16384)
            {
                FullMode = System.Threading.Channels.BoundedChannelFullMode.DropOldest
            });

            this.InitializeLogger(this.Settings, onShutdown, exitCancellationToken, synchronizationContext, setGlobally);
        }



        /// <summary>
        /// A thread-safe dictionary that stores log entries with their corresponding timestamps. The key is the timestamp when the log entry was recorded, and the value is the fully formatted log line (including the timestamp prefix).
        /// </summary>
        public readonly ConcurrentDictionary<DateTime, string> LogEntries = [];


        /// <summary>
        /// A channel that serves as a queue for log entries to be written to the log file asynchronously. Log entries are written to this channel, and a background task consumes the entries and writes them to the log file without blocking the main thread.
        /// </summary>
        private readonly System.Threading.Channels.Channel<string> _logChannel;

        // 2. Initialisierung in InitializeLogger aufrufen:
        // Startet einmalig den Background-Consumer
        private Task? _logWriterTask;
        private CancellationTokenSource? _logCts;

        /// <summary>
        /// Starts the background writer task that consumes log entries from the channel and writes them to the log file. This method should be called once during initialization to ensure that log entries are written to the file asynchronously without blocking the main thread.
        /// </summary>
        public void StartBackgroundWriter(CancellationToken cancellationToken)
        {
            if (this._logWriterTask != null)
            {
                return;
            }

            this._logCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            this._logWriterTask = Task.Run(async () =>
            {
                try
                {
                    await foreach (var line in this._logChannel.Reader.ReadAllAsync(this._logCts.Token))
                    {
                        if (this.LogFilePath != null)
                        {
                            await File.AppendAllTextAsync(this.LogFilePath, line + Environment.NewLine, this._logCts.Token);
                        }
                    }
                }
                catch (OperationCanceledException) when (this._logCts.Token.IsCancellationRequested)
                {
                    // Clean exit on cancellation
                }
            }, this._logCts.Token);
        }

        /// <summary>
        /// A counter that tracks the number of log entries in the ring buffer. This counter is used to manage the size of the log entries and ensure that the maximum number of log entries is not exceeded. When the counter reaches the maximum limit, older log entries are removed to make room for new ones.
        /// </summary>
        private int _logEntriesRingBufferCounter = 0;

        /// <summary>
        /// A thread-safe binding list that provides a chronological view of log entries for UI components. This list is updated whenever a new log entry is recorded, and it can be used to display log entries in a user interface. The list is synchronized with the UI context to ensure thread safety when updating the UI.
        /// </summary>
        public readonly BindingList<string> LogEntriesBindingList = [];

        /// <summary>
        /// A thread-safe binding list that provides a filtered view of log entries based on a specified filter phrase. This list is updated whenever a new log entry is recorded, and it can be used to display filtered log entries in a user interface. The list is synchronized with the UI context to ensure thread safety when updating the UI.
        /// </summary>
        public readonly BindingList<string> FilteredLogEntriesBindingList = [];


        /// <summary>
        /// Gets the full path of the current log file. If no log file has been created, this property will be null.
        /// </summary>
        public string? LogFilePath { get; private set; } = null;


        /// <summary>
        /// Raised whenever a new line has been recorded. The first argument is the timestamp the entry was
        /// recorded at; the second argument is the fully formatted line (including the timestamp prefix).
        /// Subscribers must tolerate being invoked from arbitrary threads.
        /// </summary>
        public event Action<DateTime, string>? LogWritten;

        /// <summary>
        /// UI synchronization context (set from the UI at startup)
        /// </summary>
        private SynchronizationContext? UiContext;

        /// <summary>
        /// Action to perform on shutdown to save logs to the repository.
        /// </summary>
        public Action? SaveToRepositoryOnShutdown { get; private set; } = null;


        /// <summary>
        /// Gets the UI synchronization context for updating the BindingList from the UI thread. If <paramref name="copy"/> is true, a copy of the synchronization context is returned; otherwise, the original context is returned. This method can be used to ensure that log entries are added to the BindingList in a thread-safe manner when updating the UI.
        /// </summary>
        /// <param name="copy">Whether to return a copy of the synchronization context.</param>
        /// <returns>The UI synchronization context or a copy of it.</returns>
        public SynchronizationContext? GetUiContext(bool copy = false) => copy ? this.UiContext?.CreateCopy() : this.UiContext;

        /// <summary>
        /// Sets the UI synchronization context for updating the BindingList from the UI thread. This method should be called from the UI thread during application startup to ensure that log entries are added to the BindingList in a thread-safe manner.
        /// </summary>
        /// <param name="context">The synchronization context of the UI thread.</param>
        public void SetUiContext(SynchronizationContext? context)
        {
            context ??= SynchronizationContext.Current;
            this.UiContext = context;
            string projectName = this.GetType().Namespace?.Split('.').FirstOrDefault() ?? "---";
            this.Log($"[Logger] RollingFileMemoryLogger UI context set for project <{projectName}>");
        }

        /// <summary>
        /// Sets the action to perform on shutdown to save logs to the repository and registers it on the
        /// provided cancellation token so it is invoked when the application is shutting down. This allows for
        /// any necessary cleanup or saving of log data.
        /// </summary>
        /// <param name="onShutdown">The action to perform on shutdown.</param>
        /// <param name="cancellationToken">Cancellation token signalling application shutdown (e.g.
        /// <c>app.Lifetime.ApplicationStopping</c>). When provided, <paramref name="onShutdown"/> is registered
        /// on it and invoked automatically on shutdown.</param>
        public void SetOnShutdownAction(Action? onShutdown, CancellationToken cancellationToken = default, bool setGlobally = false)
        {
            this.SaveToRepositoryOnShutdown = onShutdown;
            if (setGlobally)
            {
                Instance.SetOnShutdownAction(onShutdown, cancellationToken, false);
            }

            if (cancellationToken != default)
            {
                cancellationToken.Register(() => this.SaveToRepositoryOnShutdown?.Invoke());
                if (setGlobally)
                {
                    cancellationToken.Register(() => Instance.SaveToRepositoryOnShutdown?.Invoke());
                }
            }
        }


        /// <summary>
        /// Initializes the log files. If a settings object is provided, it will be used to configure the logger. Otherwise, the global <see cref="Settings"/> will be used.
        /// </summary>
        /// <param name="options">Optional settings to override the global configuration.</param>
        /// <param name="onShutdown">Optional action to perform on shutdown (e.g. save logs to repository).</param>
        public void InitializeLogger(RollingFileMemoryLoggerOptions? options = null, Action? onShutdown = null, CancellationToken? exitCancellationToken = null, SynchronizationContext? synchronizationContext = null, bool setGlobally = false)
        {
            options ??= new();
            string projectName = this.GetType().Namespace?.Split('.').FirstOrDefault() ?? "---";

            options.LogDirectory = Path.GetFullPath(options.LogDirectory ?? Path.Combine(Assembly.GetAssembly(typeof(RollingFileMemoryLogger))?.Location ?? AppContext.BaseDirectory, "Logs"));
            if (!string.IsNullOrEmpty(options.LogDirectory))
            {
                this.Settings.LogDirectory = options.LogDirectory;
                if (setGlobally)
                {
                    Instance.Settings.LogDirectory = options.LogDirectory;
                }
            }

            onShutdown ??= () => { this.SaveToRepository(); };
            this.SaveToRepositoryOnShutdown = onShutdown;
            if (setGlobally)
            {
                Instance.SaveToRepositoryOnShutdown = onShutdown;
                ApplyInstanceOnShutdownGlobally = true;
                Instance.Log($"[Logger] RollingFileMemoryLogger global shutdown action applied for project <{projectName}>");
            }

            if (exitCancellationToken.HasValue && exitCancellationToken.Value != default)
            {
                exitCancellationToken.Value.Register(() => this.SaveToRepositoryOnShutdown?.Invoke());
                if (setGlobally)
                {
                    exitCancellationToken.Value.Register(() => Instance.SaveToRepositoryOnShutdown?.Invoke());
                    ApplyInstanceOnShutdownGlobally = true;
                    Instance.Log($"[Logger] RollingFileMemoryLogger global shutdown CancellationToken registered for project <{projectName}>");
                }
            }

            if (synchronizationContext != null)
            {
                this.SetUiContext(synchronizationContext);
                if (setGlobally)
                {
                    Instance.SetUiContext(synchronizationContext);
                }
            }

            try
            {
                if (!Directory.Exists(this.Settings.LogDirectory))
                {
                    Directory.CreateDirectory(this.Settings.LogDirectory);
                }

                if (options.MaxLogFiles == 0)
                {
                    // Clear all previous logs if exactly 0 is specified
                    Directory.Delete(this.Settings.LogDirectory, true);
                    Directory.CreateDirectory(this.Settings.LogDirectory);
                }
                else if (options.MaxLogFiles >= 1)
                {
                    var existingLogs = Directory.GetFiles(this.Settings.LogDirectory, "log_*.txt")
                        .Select(path => new FileInfo(path))
                        .OrderByDescending(fi => fi.CreationTime)
                        .ToList();
                    // Keep only the most recent 'MaxLogFiles' logs
                    foreach (var oldLog in existingLogs.Skip(options.MaxLogFiles))
                    {
                        try
                        {
                            oldLog.Delete();
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error deleting old log file '{oldLog.FullName}': {ex.Message}");
                        }
                    }

                    if (setGlobally)
                    {
                        Instance.PruneOldRepositoryLogs(this.Settings.LogDirectory);
                    }
                }

                if (options.CreateLogFile)
                {
                    this.LogFilePath = Path.Combine(this.Settings.LogDirectory, $"log_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
                    File.Create(this.LogFilePath).Dispose();
                    this.Log($"Log file created at {this.LogFilePath}");
                }
            }
            catch (Exception ex)
            {
                this.Log($"Error with log files initialization: {ex.Message}");
            }
            finally
            {
                this.Log($"[Logger] RollingFileMemoryLogger initialized for project <{projectName}>");
                this.Settings = options;
                if (setGlobally)
                {
                    Instance.Settings = options;
                    ApplyInstanceSettingsGlobally = true;
                    Instance.Log($"[Logger] RollingFileMemoryLogger global settings applied for project <{projectName}>");
                }
            }
        }


        /// <summary>
        /// Logs a message with a timestamp. The message is added to the internal log entries dictionary, and if it matches the filter phrase (if any), it is also added to the filtered log entries binding list. The method raises the LogWritten event and optionally echoes the message to the console and writes it to a log file if configured.
        /// </summary>
        /// <param name="message">The message to log.</param>
        public void Log(string message)
        {
            DateTime timestamp = DateTime.Now;
            string logEntry = string.IsNullOrEmpty(this.Settings.LogTimestampFormat) ? message : $"[{timestamp.ToString(this.Settings.LogTimestampFormat)}] {message}";

            this.EnsureMaxLogEntriesWithOffloadAndBuffering();

            this.LogEntries[timestamp] = logEntry;
            if (this.Settings.Silent)
            {
                return;
            }

            if (string.IsNullOrEmpty(this.Settings.FilterPhrase) || !logEntry.Contains(this.Settings.FilterPhrase, StringComparison.OrdinalIgnoreCase))
            {
                if (this.UiContext != null)
                {
                    this.UiContext.Post(_ => this.LogEntriesBindingList.Add(logEntry), null);
                }
                else
                {
                    // Fallback: add on current thread
                    lock (this.LogEntriesBindingList)
                    {
                        this.LogEntriesBindingList.Add(logEntry);
                    }
                }
            }
            else
            {
                if (this.UiContext != null)
                {
                    this.UiContext.Post(_ => this.FilteredLogEntriesBindingList.Add(logEntry), null);
                }
                else
                {
                    // Fallback: add on current thread
                    lock (this.FilteredLogEntriesBindingList)
                    {
                        this.FilteredLogEntriesBindingList.Add(logEntry);
                    }
                }
            }

            this.RaiseLogWritten(timestamp, logEntry);

            if (this.Settings.EchoToConsole == true || this.ShouldEchoToConsole(logEntry))
            {
                Console.WriteLine(logEntry);
            }

            if (this.LogFilePath != null)
            {
                try
                {
                    // Non-blocking write to the log channel
                    this._logChannel.Writer.TryWrite(logEntry);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error writing to log file: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Logs an exception with an optional pre-text message. The exception's message and stack trace are included in the log entry. If a pre-text message is provided, it is logged before the exception details.
        /// </summary>
        /// <param name="ex">The exception to log.</param>
        /// <param name="preText">An optional pre-text message to include before the exception details.</param>
        public void Log(Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true, string? preText = null)
        {
            this.Log($"{(string.IsNullOrEmpty(preText) ? "" : preText + "\n")}Exception: {this.GetInnerExceptionsRecursively(ex, maxInnerEx)}{(appendStackTrace ? "\nStack Trace: " + ex.StackTrace : "")}");
        }

        /// <summary>
        /// Logs a contextual message together with an exception.
        /// </summary>
        /// <param name="message">The contextual message.</param>
        /// <param name="ex">The exception to append.</param>
        public void Log(string message, Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true)
        {
            this.Log($"{message} Exception: {this.GetInnerExceptionsRecursively(ex, maxInnerEx)}{(appendStackTrace ? "\nStack Trace: " + ex.StackTrace : "")}");
        }

        /// <summary>Logs an informational message.</summary>
        /// <param name="message">The message to log.</param>
        public void LogInfo(string message) => this.Log($"[INFO] {message}");

        /// <summary>Logs a success message (echoed to the console).</summary>
        /// <param name="message">The message to log.</param>
        public void LogSuccess(string message) => this.Log($"[SUCCESS] {message}");

        /// <summary>Logs a warning message (echoed to the console).</summary>
        /// <param name="message">The message to log.</param>
        public void LogWarning(string message) => this.Log($"[WARN] {message}");

        /// <summary>Logs an error message (echoed to the console).</summary>
        /// <param name="message">The message to log.</param>
        public void LogError(string message) => this.Log($"[ERROR] {message}");

        /// <summary>Logs an exception with an optional pre-text message (echoed to the console).</summary>
        /// <param name="ex">The exception to log.</param>
        /// <param name="configureAwait">Whether to configure await.</param>
        public async Task LogAsync(string message, bool? configureAwait = null)
        {
            if (configureAwait.HasValue)
            {
                await Task.Run(() => this.Log(message)).ConfigureAwait(configureAwait.Value);

            }
            else
            {
                await Task.Run(() => this.Log(message));
            }
        }

        /// <summary>Logs an exception with an optional pre-text message (echoed to the console).</summary>
        /// <param name="ex">The exception to log.</param>
        /// <param name="preText">An optional pre-text message to include before the exception details.</param>
        /// <param name="configureAwait">Whether to configure await.</param>
        public async Task LogAsync(Exception ex, int? maxInnerEx = 0, bool appendStackTrace = true, string? preText = null, bool? configureAwait = null)
        {
            if (configureAwait.HasValue)
            {
                await Task.Run(() => this.Log(ex, maxInnerEx, appendStackTrace, preText)).ConfigureAwait(configureAwait.Value);
            }
            else
            {
                await Task.Run(() => this.Log(ex, maxInnerEx, appendStackTrace, preText));
            }
        }

        /// <summary>
        /// Records a user/debugging comment anchored to the timestamp captured when the user initiated it
        /// (for example when a "comment now" button was pressed), so it lands in the log at the right moment.
        /// </summary>
        /// <param name="capturedAt">The timestamp captured at the moment the comment was initiated.</param>
        /// <param name="comment">The free-form comment text.</param>
        public void AddComment(DateTime? capturedAt = null, TimeSpan? elapsedSince = null, string comment = "<!!!>")
        {
            if (string.IsNullOrWhiteSpace(comment))
            {
                return;
            }

            elapsedSince ??= TimeSpan.Zero;
            capturedAt ??= DateTime.Now.Subtract(elapsedSince.Value);

            string logEntry = string.IsNullOrEmpty(this.Settings.LogTimestampFormat) ? $"[COMMENT] {comment}" : $"[{capturedAt.Value.ToString(this.Settings.LogTimestampFormat.VerifyFormatString(out var _))}] [COMMENT] {comment}";

            this.LogEntries[capturedAt.Value] = logEntry;

            if (this.UiContext != null)
            {
                this.UiContext.Post(_ => this.LogEntriesBindingList.Add(logEntry), null);
            }
            else
            {
                lock (this.LogEntriesBindingList)
                {
                    this.LogEntriesBindingList.Add(logEntry);
                }
            }

            this.RaiseLogWritten(capturedAt.Value, logEntry);

            if (this.LogFilePath != null)
            {
                try
                {
                    File.AppendAllText(this.LogFilePath, logEntry + Environment.NewLine);
                }
                catch
                {
                }
            }
        }

        /// <summary>
        /// Asynchronously records a user/debugging comment anchored to the timestamp captured when the user initiated it (for example when a "comment now" button was pressed), so it lands in the log at the right moment.
        /// </summary>
        /// <param name="capturedAt">The timestamp captured at the moment the comment was initiated.</param>
        /// <param name="elapsedSince">The elapsed time since the comment was captured.</param>
        /// <param name="comment">The free-form comment text.</param>
        /// <param name="timeStampFormat">The format string for the timestamp.</param>
        /// <param name="configureAwait">Whether to configure await.</param>
        /// <returns>A task that represents the asynchronous operation.</returns>
        public async Task AddCommentAsync(DateTime? capturedAt = null, TimeSpan? elapsedSince = null, string comment = "<!!!>", bool? configureAwait = null)
        {
            if (configureAwait.HasValue)
            {
                await Task.Run(() => this.AddComment(capturedAt, elapsedSince, comment)).ConfigureAwait(configureAwait.Value);
            }
            else
            {
                await Task.Run(() => this.AddComment(capturedAt, elapsedSince, comment));
            }
        }


        /// <summary>
        /// Returns a read-only list of log lines, optionally filtered based on the <see cref="FilterPhrase"/>. If <paramref name="returnFilteredLog"/> is true, only the filtered log entries are returned; if false, only the main log list is returned; if null, all log entries are returned in chronological order.
        /// </summary>
        /// <param name="returnFilteredLog">If true, returns ONLY the filtered log entries, if null, returns all log entries chronologically merged, if false, returns only the main log list.</param>
        /// <param name="reverseOrder">If true, returns the log entries in reverse chronological order.</param>
        /// <returns></returns>
        public IReadOnlyList<string> GetLogLines(bool? returnFilteredLog = false, bool reverseOrder = false)
        {
            var result = returnFilteredLog switch
            {
                true => this.FilteredLogEntriesBindingList,
                false => this.LogEntriesBindingList,
                null => this.LogEntries.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value),
            };

            return reverseOrder ? result.Reverse().ToList() : result.ToList();
        }


        /// <summary>
        /// Returns the full paths of all log files in the log directory, ordered by creation time (newest first). This method searches for files with the extensions ".txt" and ".log" in the specified log directory and returns their paths as an enumerable collection. The returned list can be used to access or manage existing log files.
        /// </summary>
        /// <returns>An enumerable collection of full paths to log files.</returns>
        public string[] GetAllLogFilePaths()
        {
            return Directory.GetFiles(this.Settings.LogDirectory, "*.txt").Concat(Directory.GetFiles(this.Settings.LogDirectory, "*.log"))
                .OrderByDescending(f => f)
                .ToArray();
        }

        /// <summary>
        /// Returns the full path of a previous log file based on the specified index. The index is zero-based, where 0 corresponds to the most recent log file, 1 corresponds to the second most recent log file, and so on. If the specified index is out of range (i.e., there are fewer log files than the index), this method returns null.
        /// </summary>
        /// <param name="backIndex">The zero-based index of the log file to retrieve, where 0 is the most recent log file.</param>
        /// <returns>The full path of the previous log file, or null if the index is out of range.</returns>
        public string? GetPreviousLogFilePath(int backIndex = 0)
        {
            return this.GetAllLogFilePaths().Select(l => new FileInfo(l)).OrderByDescending(f => f.CreationTime) is IEnumerable<FileInfo> fileInfos ? fileInfos.Count() > backIndex ? fileInfos.ElementAt(backIndex).FullName : null : null;
        }


        /// <summary>
        /// Saves the current log entries to a timestamped TXT file in the repository's log directory.
        /// </summary>
        /// <param name="differentFilePathOrDirectory">An optional different file path or directory to save the log file to. If null, the default repository log directory is used.</param>
        /// <param name="defaultFileName">The default file name to use for the log file.</param>
        /// <param name="timeStampFormat">The format to use for the timestamp in the file name.</param>
        /// <param name="extension">The file extension to use for the log file.</param>
        /// <param name="returnFilteredLog">Whether to return the filtered log entries.</param>
        /// <param name="reverseOrder">Whether to reverse the order of the log entries.</param>
        /// <param name="forceSave">Whether to force saving the log file even if the setting to save to the repository is disabled.</param>
        /// <returns>The full path of the saved log file, or an empty string if the log file was not saved.</returns>
        public string SaveToRepository(string? differentFilePathOrDirectory = null, bool? returnFilteredLog = null, bool reverseOrder = false, bool forceSave = false)
        {
            if (!this.Settings.SaveToRepository && !forceSave)
            {
                return string.Empty;
            }

            string path = this.ResolveRepositoryLogFilePath(differentFilePathOrDirectory ?? this.Settings.SaveToRepositoryCustomFilePath);
            string fileName = Path.GetFileName(path) ?? (string.IsNullOrEmpty(this.Settings.LogFileBaseName) ? "dotnet-Application_Log_" : this.Settings.LogFileBaseName);
            string directory = Path.GetDirectoryName(path) ?? this.ResolveRepositoryDirectory();

            IReadOnlyList<string> snapshot = this.GetLogLines(returnFilteredLog, reverseOrder);

            var sb = new StringBuilder();
            sb.AppendLine("==============================================================");
            sb.AppendLine(UniversalHelper.SanitizeString(fileName, null, "".ToCharArray(), " ", true, true) + " Log");

            string timestamp = DateTime.Now.ToString(this.Settings.FileTimestampFormat);

            nint logCount = new(snapshot.LongCount());
            sb.AppendLine($"Saved at : {timestamp}");
            sb.Append("Entries   : ").AppendLine(logCount.ToString("N0"));
            sb.AppendLine("==============================================================");
            sb.AppendLine();
            sb.AppendJoin(Environment.NewLine, snapshot);

            this.PruneOldRepositoryLogs(directory);
            File.WriteAllText(path, sb.ToString());

            this.Log($"[SUCCESS] Log saved to {Path.GetFileName(path)} ({logCount:N0} entries)");

            return path;
        }

        /// <summary>
        /// Configures the logger to save all recorded log lines to a timestamped TXT file under the repository's <c>AsynCUDA13.Shared\Logs</c> folder.
        /// </summary>
        /// <param name="configureToggle">If true, enables saving to the repository; if false, disables it; if null, toggles the current state.</param>
        /// <param name="subDirOrDifferentPath">Custom subdirectory or different path for the log file.</param>
        /// <param name="maxPreviousLogFiles">Maximum number of previous log files to retain.</param>
        /// <param name="onShutdown">Action to perform on shutdown.</param>
        /// <param name="logFilesBaseName">Base name for the log files.</param>
        /// <param name="timeStampFormat">Timestamp format to append to the file name.</param>
        /// <param name="extension">File extension for the log file.</param>
        /// <returns>The full path of the resolved log file.</returns>
        public string ConfigureSaveToRepository(bool? configureToggle = null, string? subDirOrDifferentPath = null, int maxPreviousLogFiles = 8, Action? onShutdown = null)
        {
            this.Settings.SaveToRepository = configureToggle == null ? !this.Settings.SaveToRepository : configureToggle.Value;
            this.SaveToRepositoryOnShutdown = onShutdown;
            this.Settings.SaveToRepositoryCustomFilePath = string.IsNullOrEmpty(subDirOrDifferentPath) ? null : this.ResolveRepositoryLogFilePath(subDirOrDifferentPath);

            this.Settings.MaxRepositoryLogFiles = Math.Clamp(maxPreviousLogFiles, 0, int.MaxValue);
            this.PruneOldRepositoryLogs();

            try
            {
                if (this.Settings.SaveToRepository)
                {
                    this.SaveToRepository(subDirOrDifferentPath);
                }
            }
            catch (Exception ex)
            {
                this.Log($"Error configuring save to repository: {ex.Message}");
            }

            return this.Settings.SaveToRepositoryCustomFilePath ?? this.ResolveRepositoryDirectory();
        }


        /// <summary>
        /// Returns an array of namespace strings for the specified project name. If the project name is null, the namespace of the StaticLogger class is used.
        /// </summary>
        /// <param name="projectName">The name of the project to retrieve namespaces for. If null, the namespace of the StaticLogger class is used.</param>
        /// <param name="includeSubNamespaces">If true, includes sub-namespaces of the specified project namespace.</param>
        /// <param name="ignoreCase">If true, performs a case-insensitive comparison of namespaces.</param>
        /// <returns>An array of namespace strings, or null if the project name is not specified and the StaticLogger namespace is not available.</returns>
        public string[]? GetNamespacesForProject(string? projectName = null, bool includeSubNamespaces = true, bool ignoreCase = true)
        {
            if (string.IsNullOrEmpty(projectName))
            {
                projectName = typeof(RollingFileMemoryLogger).Namespace;
                if (string.IsNullOrEmpty(projectName))
                {
                    return null;
                }
            }

            var assembly = Assembly.GetExecutingAssembly();
            return assembly.GetTypes()
                .Where(t => t.IsClass && t.Namespace != null && (includeSubNamespaces ? t.Namespace.StartsWith(projectName) : t.Namespace.Equals(projectName, (ignoreCase ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))))
                .Select(t => t.Namespace)
                .Distinct()
                .OfType<string>()
                .ToArray();
        }

        /// <summary>
        /// Resolves the full path of the repository or project directory based on the provided project name and optional subdirectory. If the project name is null, the namespace of the StaticLogger class is used.
        /// </summary>
        /// <param name="projectName">Project name within the repository, string.Empty ("") for root path, null for this namespaces project.</param>
        /// <param name="subPath">Optional relative subdirectory within the project or repository directory.</param>
        /// <param name="ensureDirectoryExists">If true, ensures that the resolved directory exists by creating it if necessary.</param>
        /// <returns>The full path to the resolved repository or project directory.</returns>
        public string ResolveRepositoryDirectory(string? projectName = null, string subPath = "Logs", bool ensureDirectoryExists = false)
        {
            // Auto-namespace with this class if null, otherwise use the provided project name (string.Empty for root path)
            if (projectName is null)
            {
                projectName = typeof(RollingFileMemoryLogger).Namespace;
            }
            else if (!string.IsNullOrEmpty(projectName))
            {
                projectName = this.GetNamespacesForProject(projectName, false)?.FirstOrDefault();
            }

            string path = Path.GetFullPath(AppContext.BaseDirectory);
            DirectoryInfo? dir = new(path);
            string? assemblyName = Assembly.GetExecutingAssembly().GetName() is AssemblyName assembly ? assembly.Name ?? assembly.FullName : null;
            while (dir != null)
            {
                if (File.Exists(Path.Combine(dir.FullName, $"{assemblyName}.sln")) || File.Exists(Path.Combine(dir.FullName, $"{assemblyName}.slnx")))
                {
                    path = Directory.GetFiles(dir.FullName, $"{assemblyName}.sln").FirstOrDefault() ?? Directory.GetFiles(dir.FullName, $"{assemblyName}.slnx").FirstOrDefault() ?? string.Empty;
                    if (!string.IsNullOrEmpty(projectName))
                    {
                        var options = new EnumerationOptions
                        {
                            MatchCasing = MatchCasing.CaseInsensitive,
                            RecurseSubdirectories = true
                        };
                        path = Directory.EnumerateFiles(dir.FullName, $"{projectName}.csproj", options).FirstOrDefault() ?? string.Empty;
                    }
                }

                dir = dir.Parent;
            }

            string fullPath = Path.GetFullPath(path);
            if (!string.IsNullOrEmpty(fullPath) && !Directory.Exists(fullPath))
            {
                Directory.CreateDirectory(fullPath);
            }

            return fullPath;
        }

        /// <summary>
        /// Resolves the full path of a log file in the repository's log directory, optionally using a custom file path or directory. If a custom file path is provided and it points to an existing directory, the log file will be created in that directory. If the custom path points to a non-existing directory, the log file will be created in the repository's default log directory.
        /// </summary>
        /// <param name="differentFilePathOrDirectory">Custom file path or directory for the log file.</param>
        /// <param name="defaultFileName">Default file name to use if a custom file name is not provided.</param>
        /// <param name="timeStampFormat">Timestamp format to append to the file name.</param>
        /// <param name="extension">File extension for the log file.</param>
        /// <returns>The full path of the resolved log file.</returns>
        public string ResolveRepositoryLogFilePath(string? differentFilePathOrDirectory = null)
        {
            string repoRootDir = this.ResolveRepositoryDirectory();
            string directory;
            if (Directory.Exists(differentFilePathOrDirectory))
            {
                directory = differentFilePathOrDirectory;
            }
            else
            {
                directory = this.ResolveRepositoryDirectory();
                Directory.CreateDirectory(directory);
            }

            string fileName;
            if (!string.IsNullOrEmpty(differentFilePathOrDirectory) && !Directory.Exists(differentFilePathOrDirectory))
            {
                fileName = Path.GetFileName(differentFilePathOrDirectory);
                if (string.IsNullOrEmpty(fileName))
                {
                    fileName = this.Settings.LogFileBaseName;
                }
            }
            else
            {
                fileName = this.Settings.LogFileBaseName;
            }

            try
            {
                fileName = $"{fileName}{DateTime.Now.ToString(this.Settings.FileTimestampFormat)}";
            }
            catch
            {
                fileName = $"{fileName}{DateTime.Now:yyyy-MM-dd_HH-mm-ss}";
            }

            return Path.GetFullPath(Path.Combine(directory, fileName + "." + this.Settings.LogFileExtension.Trim('.')));
        }


        /// <summary>
        /// Returns a string representation of all inner exceptions of the provided exception, including their messages and stack traces, recursively. This method is useful for logging or displaying detailed information about nested exceptions.
        /// </summary>
        /// <param name="ex">The exception to process.</param>
        /// <param name="openingBracket">The opening bracket to use for inner exception messages.</param>
        /// <param name="closingBracket">The closing bracket to use for inner exception messages.</param>
        /// <param name="separator">The separator to use between inner exception messages.</param>
        /// <returns>A string containing the details of the exception and all its inner exceptions.</returns>
        public string GetInnerExceptionsRecursively(Exception ex, int? maxDepth = null, bool appendStackTrace = true, string openingBracket = "(", string closingBracket = ")", string separator = " ", bool asSingleLine = false)
        {
            if (ex == null)
            {
                return string.Empty;
            }

            if (maxDepth <= 0)
            {
                return ex.Message;
            }

            StringBuilder sb = new();
            sb.AppendLine($"Exception: {ex.GetType().FullName}");
            string message = $"Message: {ex.Message}";

            Exception? inner = ex.InnerException;
            int count = 0;
            while (inner != null)
            {
                message += $"{separator}{openingBracket}{inner.Message}";
                inner = inner.InnerException;
                count++;
                if (maxDepth.HasValue && count >= maxDepth.Value)
                {
                    break;
                }
            }
            message += string.Concat(Enumerable.Repeat(closingBracket, count));

            sb.AppendLine(message);
            if (appendStackTrace)
            {
                sb.AppendLine($"StackTrace: {ex.StackTrace}");
            }
            return sb.Replace(Environment.NewLine, asSingleLine ? " " : Environment.NewLine).ToString();
        }

        /// <summary>
        /// Returns a string representation of all inner exceptions of the provided exception, including their messages and stack traces, recursively. This method is useful for logging or displaying detailed information about nested exceptions. It uses default formatting with parentheses and spaces to separate inner exception messages.
        /// </summary>
        /// <param name="ex">The exception to process.</param>
        /// <returns>A string containing the details of the exception and all its inner exceptions.</returns>
        public string GetInnerExceptionsRecursively(Exception ex)
        {
            return this.GetInnerExceptionsRecursively(ex, this.Settings.ExceptionPrintSettings.InnerExceptionMaxDepth, this.Settings.ExceptionPrintSettings.InnerExceptionAppendStackTrace, this.Settings.ExceptionPrintSettings.InnerExceptionOpeningBracket, this.Settings.ExceptionPrintSettings.InnerExceptionClosingBracket, this.Settings.ExceptionPrintSettings.InnerExceptionSeparator, this.Settings.ExceptionPrintSettings.InnerExceptionAsSingleLine);
        }



        /// <summary>
        /// Clears all recorded log entries from the internal dictionary and the binding lists. This method is thread-safe and ensures that the UI context is used to update the binding lists if available. After calling this method, both <see cref="LogEntriesBindingList"/> and <see cref="FilteredLogEntriesBindingList"/> will be empty.
        /// </summary>
        public void ClearLogs()
        {
            this.LogEntries.Clear();
            if (this.UiContext != null)
            {
                this.UiContext.Post(_ => this.LogEntriesBindingList.Clear(), null);
                this.UiContext.Post(_ => this.FilteredLogEntriesBindingList.Clear(), null);
            }
            else
            {
                lock (this.LogEntriesBindingList)
                {
                    this.LogEntriesBindingList.Clear();
                }
                lock (this.FilteredLogEntriesBindingList)
                {
                    this.FilteredLogEntriesBindingList.Clear();
                }
            }
        }

        /// <summary>
        /// Deletes the oldest saved log files so at most <see cref="MaxRepositoryLogFiles"/> remain.
        /// </summary>
        private void PruneOldRepositoryLogs(string? directory = null)
        {
            directory ??= this.Settings.SaveToRepositoryCustomFilePath is not null ? Path.GetDirectoryName(this.Settings.SaveToRepositoryCustomFilePath) ?? this.ResolveRepositoryDirectory() : this.ResolveRepositoryDirectory();

            try
            {
                FileInfo[] files = new DirectoryInfo(directory)
                    .GetFiles("AggregatedLog_*.txt")
                    .OrderByDescending(f => f.LastWriteTimeUtc)
                    .ToArray();

                foreach (FileInfo file in files.Skip(this.Settings.MaxRepositoryLogFiles))
                {
                    try
                    {
                        file.Delete();
                    }
                    catch
                    {
                    }
                }
            }
            catch
            {
            }
        }


        /// <summary>
        /// Determines whether a formatted line should be echoed to the console. Per the project's CLI
        /// logging guideline, only success, error and warning lines are printed.
        /// </summary>
        private bool ShouldEchoToConsole(string logEntry)
        {
            if (this.Settings.Silent)
            {
                return false;
            }

            if (this.Settings.EchoToConsole == true)
            {
                return true;
            }
            else if (this.Settings.EchoToConsole == false)
            {
                return false;
            }
            else
            {
                return this.Settings.EchoToConsoleKeyPhrases.Any(phrase => logEntry.Contains(phrase, StringComparison.OrdinalIgnoreCase));
            }
        }

        /// <summary>
        /// Safely raises <see cref="LogWritten"/>, swallowing any subscriber exception so logging never fails.
        /// </summary>
        private void RaiseLogWritten(DateTime timestamp, string line)
        {
            try
            {
                LogWritten?.Invoke(timestamp, line);
                this._logEntriesRingBufferCounter++;
            }
            catch
            {
            }
        }

        /// <summary>
        /// Ensures that the number of log entries does not exceed the maximum allowed, and if it does, it either removes the oldest entries (if using a ring buffer) or saves the current log to a file and clears the log entries. This method handles different configurations for log entry management based on the settings provided.
        /// </summary>
        private void EnsureMaxLogEntriesWithOffloadAndBuffering()
        {
            int maxVal = this.Settings.MaxLogEntries ?? int.MaxValue;
            if (maxVal <= this.LogEntries.Count)
            {
                bool? error = null;
                switch (this.Settings.UseRingBuffer)
                {
                    case true:
                        {
                            while (maxVal <= this.LogEntries.Count)
                            {
                                var oldestKeyOpt = this.LogEntries.Keys.OrderBy(k => k).FirstOrDefault();
                                if (oldestKeyOpt != default)
                                {
                                    error = !this.LogEntries.TryRemove(oldestKeyOpt, out _);
                                }
                                else
                                {
                                    break;
                                }
                            }
                            break;
                        }
                    case false:
                        {
                            this.SaveToRepository();
                            this.LogEntries.Clear();
                            this.LogEntriesBindingList.Clear();
                            this.FilteredLogEntriesBindingList.Clear();
                            int? fileIndex = int.TryParse(Path.GetFileNameWithoutExtension(this.LogFilePath)?.Split('_').Last().Trim(), out var index) ? index : null;
                            this.LogFilePath = Path.GetFullPath(this.LogFilePath?.TrimEnd('_') + "_" + (fileIndex.HasValue ? (fileIndex.Value + 1).ToString() : "0"));
                            break;
                        }
                    case null:
                        {
                            while (maxVal <= this.LogEntries.Count)
                            {
                                var oldestEntry = this.LogEntries.Keys.OrderBy(k => k).FirstOrDefault();
                                if (oldestEntry != default)
                                {
                                    error = !this.LogEntries.TryRemove(oldestEntry, out _);
                                }
                                else
                                {
                                    break;
                                }
                            }
                            if (this._logEntriesRingBufferCounter >= maxVal)
                            {
                                this.SaveToRepository();
                                this._logEntriesRingBufferCounter = 0;
                                int? fileIndexNull = int.TryParse(Path.GetFileNameWithoutExtension(this.LogFilePath)?.Split('_').Last().Trim(), out var indexNull) ? indexNull : null;
                                this.LogFilePath = Path.GetFullPath(this.LogFilePath?.TrimEnd('_') + "_" + (fileIndexNull.HasValue ? (fileIndexNull.Value + 1).ToString() : "0"));
                            }
                            break;
                        }
                }
                if (error == true)
                {
                    this.LogWarning($"[WARN] Failed to remove the oldest log entry while ensuring max log entries. Current count: {this.LogEntries.Count}, Max allowed: {maxVal}");
                }
            }
        }

    }
}