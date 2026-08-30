using ParallelOnnxDeltaForge.Shared.MediaDtos;
using ParallelOnnxDeltaForge.Media;

namespace ParallelOnnxDeltaForge.Api.Services
{
    public interface IAssetProvider
    {
        ImageObj? GetImage(Guid id);
        ImageObj? GetImage(string name);
        AudioObj? GetAudio(Guid id);
        AudioObj? GetAudio(string name);

        ImageInfo GetImageInfo(ImageObj image);
        AudioInfo GetAudioInfo(AudioObj audio);
        ImageInfo? GetImageInfo(Guid imageId);
        AudioInfo? GetAudioInfo(Guid audioId);

        AudioObj? CreateFromInfo(AudioInfo info, bool tryAdd = true, bool disposeIfFailedToAdd = true, bool emptyData = false, long? pointer = 0);
        ImageObj? CreateFromInfo(ImageInfo info, bool tryAdd = true, bool disposeIfFailedToAdd = true, bool emptyData = false, long? pointer = 0);

        Guid? VerifyAssetId(Guid assetId);
        Guid[] VerifyAssetIds(IEnumerable<Guid> ids);

        Guid? GetAssetIdByPointer(long pointer);
        Guid[] GetAssetIdsByPointers(IEnumerable<long> pointers);
    }
}