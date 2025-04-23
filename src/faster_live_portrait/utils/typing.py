# -*- coding: utf-8 -*-
from typing import TypedDict, List
import torch

# Type hint for the output of FaceAnalysisModel.predict
# Contains tensors for bounding box, score, and initial landmarks
class LandmarkResultDict(TypedDict):
    bbox: torch.Tensor  # Shape: (4,) [x1, y1, x2, y2]
    score: torch.Tensor # Shape: (1,)
    landmark: torch.Tensor # Shape: (K, 2 or 3)
    # kps: torch.Tensor | None # Optional initial keypoints from detector

# Alias for clarity, though currently structure is the same
FaceAnalysisResultDict = LandmarkResultDict

# Potentially add more specific types later if needed
# e.g., for motion info, config dicts, etc. 