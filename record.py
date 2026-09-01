"""FishRecord: the one shared input shape for every measurement method."""

from dataclasses import dataclass, field


@dataclass
class FishRecord:
    image_name: str
    points: dict[int, tuple[float, float]]
    px_per_inch: float | None
    estimated_length_in: float | None
    depth: dict[int, float] | None = None
    camera_intrinsic: list | None = None
    camera_transform: list | None = None
    estimated_length_3d_in: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def has_3d(self) -> bool:
        return self.depth is not None and self.camera_intrinsic is not None \
               and self.camera_transform is not None
