using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using ParallelOnnxDeltaForge.Shared;
using ParallelOnnxDeltaForge.Shared.Dtos;
using ParallelOnnxDeltaForge.Shared.Interfaces;

namespace ParallelOnnxDeltaForge.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DeltaForgeController : ApiControllerBase
    {
        private readonly IOnnxDeltaForgeService _forge;

        public DeltaForgeController(IOnnxDeltaForgeService forge) : base()
        {
            this._forge = forge;
        }

        [HttpPost("load-model")]
        public async Task<ActionResult<Guid>> LoadModelAsync([FromBody] LoadModelRequest req)
        {
            try
            {
                var id = await this._forge.LoadModelAsync(req.ModelPath, req.DeviceId);
                return this.Ok(id);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Load model error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Load failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpPost("load-model-with-lora")]
        public async Task<ActionResult<Guid>> LoadModelWithLoraAsync([FromBody] LoadModelLoraRequest req)
        {
            try
            {
                var id = await this._forge.LoadModelWithLoraAsync(
                    req.ModelPath, req.LoraPath, req.LoraName, req.Rank, req.ScaleFactor, req.DeviceId);
                return this.Ok(id);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Load model+LoRA error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Load failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpPost("load-lora-adapter")]
        public async Task<ActionResult<LoraAdapterInfo>> LoadLoraAdapterAsync([FromBody] LoadLoraRequest req)
        {
            try
            {
                var info = await this._forge.LoadLoraAdapterAsync(req.AdapterPath, req.Name, req.Rank, req.ScaleFactor);
                return this.Ok(info);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Load LoRA error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Load failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpPost("infer")]
        public async Task<ActionResult<InferenceResponse>> RunInferenceAsync([FromBody] InferenceRequest req)
        {
            try
            {
                var response = await this._forge.RunInferenceAsync(req);
                return this.Ok(response);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Inference error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Inference failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpPost("compute-deltas")]
        public async Task<ActionResult<LoRADeltaSet>> ComputeDeltasAsync([FromBody] ComputeDeltasRequest req)
        {
            try
            {
                var deltas = await this._forge.ComputeDeltasAsync(req.TargetRank);
                return this.Ok(deltas);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Delta computation error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Computation failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpPost("export-deltas")]
        public async Task<ActionResult<DeltaExportResult>> ExportDeltasAsync([FromBody] ParallelOnnxDeltaForge.Shared.Dtos.ExportDeltasRequest req)
        {
            try
            {
                var deltas = await this._forge.ComputeDeltasAsync(req.TargetRank);
                var result = await this._forge.ExportDeltasAsync(deltas, req.Mode, req.OutputPath);
                return this.Ok(result);
            }
            catch (Exception ex)
            {
                RollingFileMemoryLogger.Instance.LogError($"[DeltaForge] Export error: {ex.Message}");
                return this.StatusCode(500, new ProblemDetails { Title = "Export failed", Detail = ex.Message, Status = 500 });
            }
        }

        [HttpGet("context")]
        public ActionResult<IReadOnlyList<ContextTurn>> GetContext()
        {
            return this.Ok(this._forge.GetContext());
        }

        [HttpGet("adapters")]
        public ActionResult<IReadOnlyList<LoraAdapterInfo>> GetAdapters()
        {
            return this.Ok(this._forge.GetLoadedAdapters());
        }

        [HttpPost("clear-context")]
        public async Task<ActionResult> ClearContextAsync()
        {
            await this._forge.ClearContextAsync();
            return this.NoContent();
        }

        [HttpDelete("unload")]
        public ActionResult UnloadModel([FromQuery] Guid? sessionId = null)
        {
            this._forge.UnloadModel(sessionId);
            return this.NoContent();
        }

        [HttpDelete("unload-all")]
        public ActionResult UnloadAll()
        {
            this._forge.UnloadAll();
            return this.NoContent();
        }
    }
}
