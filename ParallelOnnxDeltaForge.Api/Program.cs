using Microsoft.AspNetCore.SignalR;
using Newtonsoft.Json;
using ParallelOnnxDeltaForge.Api.Services;
using ParallelOnnxDeltaForge.Media;
using ParallelOnnxDeltaForge.Runtime;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Interfaces;
using ParallelOnnxDeltaForge.Shared.Options;
using Swashbuckle.AspNetCore.SwaggerUI;
using static ParallelOnnxDeltaForge.Shared.RollingFileMemoryLogger;

namespace ParallelOnnxDeltaForge.Api
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Get StaticLoggerSettings from configuration
            var loggerSettings = builder.Configuration.GetSection("LoggerSettings").Get<RollingFileMemoryLoggerOptions>() ?? new();

            // Add services to the container.
            builder.Services.AddSingleton<IRollingFileMemoryLogger, RollingFileMemoryLogger>();
            builder.Services.AddSingleton<IMediaCollection, AudioCollection>();
            builder.Services.AddSingleton<IMediaCollection, ImageCollection>();
            builder.Services.AddSingleton<IAssetProvider, AssetProvider>();

            // DeltaForge LoRA pipeline
            builder.Services.AddSingleton<ParallelOnnxDeltaForge.Runtime.LoRAAdapterLoader>();
            builder.Services.AddSingleton<ParallelOnnxDeltaForge.Runtime.ContextTracker>();
            builder.Services.AddSingleton<ParallelOnnxDeltaForge.Runtime.LoRADeltaComputationService>();
            builder.Services.AddSingleton<ParallelOnnxDeltaForge.Runtime.DeltaExporter>();
            builder.Services.AddSingleton<IOnnxGpuService, OnnxGpuService>();
            builder.Services.AddSingleton<IOnnxDeltaForgeService, ParallelOnnxDeltaForge.Runtime.OnnxDeltaForgeService>();

            // Configure CORS to allow requests from the specified origins
            builder.Services.AddCors(options =>
            {
                options.AddPolicy("WebApp", policy =>
                {
                    var origins = builder.Configuration.GetSection("CORS:AllowedOrigins").Get<string[]>() ?? [];
                    policy.WithOrigins(origins)
                        .AllowAnyMethod()
                        .AllowAnyHeader()
                        .AllowCredentials();
                });
            });

            // Configure JSON serialization options to handle unknown types as JsonNode
            builder.Services.AddControllers()
                            .AddJsonOptions(options =>
                            {
                                options.JsonSerializerOptions.UnknownTypeHandling = System.Text.Json.Serialization.JsonUnknownTypeHandling.JsonNode;
                            });

            // Add swagger and SignalR services
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSignalR(options =>
            {
                options.EnableDetailedErrors = true;
            });
            builder.Services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new() { Title = $"P-ONNX-ẟ-Forge v1", Version = "v1" });
            });

            var app = builder.Build();

            // Initialize the StaticLogger with the settings and configure it to save logs on application stopping
            Instance.InitializeLogger(loggerSettings, () => Instance.SaveToRepository(), app.Lifetime.ApplicationStopping, SynchronizationContext.Current);

            // Configure the HTTP(S) request pipeline.
            app.UseHttpsRedirection();
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI(options =>
                {
                    options.SwaggerEndpoint("/swagger/v1/swagger.json", $"P-ONNX-ẟ-Forge v1 v1");
                    options.RoutePrefix = "swagger";
                });
            }

            // App CORS and Authorization middleware
            app.UseCors("WebApp");
            app.UseAuthorization();

            // Initialize LogBroadcaster with the HubContext
            var hubContext = app.Services.GetRequiredService<IHubContext<Hubs.LogHub>>();
            Hubs.LogBroadcaster.SetHubContext(hubContext);
            Hubs.LogBroadcaster.SubscribeToLogger();

            // Map controllers and SignalR hub endpoints
            app.MapControllers();
            app.MapHub<Hubs.LogHub>("/logHub")
                .RequireCors("WebApp");

            app.Run();
        }
    }
}
