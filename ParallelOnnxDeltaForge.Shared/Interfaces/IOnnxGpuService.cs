using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ParallelOnnxDeltaForge.Shared.Interfaces
{
    public interface IOnnxGpuService : IDisposable
    {
        /// <summary>
        /// Lists all available CUDA devices by their IDs on the system.
        /// </summary>
        IReadOnlyList<int> GetAvailableCudaDevices();

        /// <summary>
        /// Asynchronously loads an ONNX model onto a specified GPU device and returns a unique session ID (Guid) for the loaded model. This method ensures that only one model is loaded onto the GPU at a time, preventing conflicts and ensuring thread safety. The session ID can be used to manage the model's lifecycle, including unloading it when no longer needed.
        /// </summary>
        /// <param name="modelPath">The physical path to the .onnx/.onnxdata file.</param>
        /// <param name="deviceId">The target GPU (e.g., 0 for primary, 1 for secondary).</param>
        /// <returns>A unique session ID for the loaded model.</returns>
        Task<Guid> LoadModelAsync(string modelPath, int deviceId);

        /// <summary>
        /// Unloads a model and frees the VRAM as well as unmanaged resources.
        /// </summary>
        void UnloadModel(Guid? sessionId);

        /// <summary>
        /// Unloads all models and clears the VRAM.
        /// </summary>
        void UnloadAll();
    }
}