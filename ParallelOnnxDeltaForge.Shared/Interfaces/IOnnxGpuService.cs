using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ParallelOnnxDeltaForge.Shared.Interfaces
{
    /// <summary>
    /// A service class that manages ONNX model loading and inference on GPU devices, ensuring thread safety and proper resource management. It allows loading models onto specified GPU devices, unloading them to free resources, and provides information about available CUDA devices.
    /// </summary>
    public interface IOnnxGpuService : IDisposable
    {
        /// <summary>
        /// Retrieves a read-only list of available CUDA devices on the system. This method provides information about the GPU devices that can be used for ONNX model inference.
        /// </summary>
        /// <returns>A read-only list of available CUDA device IDs.</returns>
        IReadOnlyList<int> GetAvailableCudaDevices();

        /// <summary>
        /// Asynchronously loads an ONNX model onto a specified GPU device. This method ensures that only one model is loaded onto the GPU at a time by using a semaphore for thread safety.
        /// </summary>
        /// <param name="modelPath">File path to the ONNX model.</param>
        /// <param name="deviceId">The ID of the GPU device on which to load the model.</param>
        /// <returns>The unique session ID (Guid) for the loaded model.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the specified model file does not exist.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the specified device ID is invalid (not in the list of available CUDA devices).</exception>
        Task<Guid> LoadModelAsync(string modelPath, int deviceId);

        /// <summary>
        /// Unloads an ONNX model associated with the specified session ID, freeing up GPU VRAM. If the session ID does not exist, a warning is logged.
        /// </summary>
        /// <param name="sessionId">The unique session ID (Guid) of the model to unload. If null, the first active session will be unloaded.</param>
        void UnloadModel(Guid? sessionId);

        /// <summary>
        /// Unloads all active ONNX models, freeing up GPU VRAM for each session. This method iterates through all active sessions and unloads them, ensuring that all resources are properly disposed of.
        /// </summary>
        void UnloadAll();
    }
}