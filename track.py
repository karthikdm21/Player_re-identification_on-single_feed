from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class Track:
    id: int
    bbox: Tuple[int, int, int, int]
    color_hist: np.ndarray
    last_seen: int
    confidence: float
    stabilized: bool = False
    frame_count: int = 0
    exit_position: Optional[str] = None

    def __post_init__(self):
        self.center = ((self.bbox[0] + self.bbox[2]) // 2, 
                       (self.bbox[1] + self.bbox[3]) // 2)
