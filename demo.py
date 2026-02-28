import torch

from crowd_density import OptimizedSpatialCrowdNet, TemporalSpatialCrowdNet


def main() -> None:
    temporal_model = TemporalSpatialCrowdNet(use_pretrained_backbone=False).eval()
    spatial_model = OptimizedSpatialCrowdNet().eval()

    video = torch.randn(2, 4, 3, 224, 224)
    frame = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        temporal_density = temporal_model(video)
        spatial_density = spatial_model(frame)

    print("Temporal model output:", temporal_density.shape)
    print("Spatial model output:", spatial_density.shape)


if __name__ == "__main__":
    main()
