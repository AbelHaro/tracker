"""
Object tracking with a shared NumPy interface.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['BoxFormat', 'ByteTracker', 'Detection', 'Prediction', 'Sort', 'Tracker', 'add']
class BoxFormat:
    """
    Members:
    
      TLWH
    
      CXCYWH
    
      XYXY
    """
    CXCYWH: typing.ClassVar[BoxFormat]  # value = <BoxFormat.CXCYWH: 1>
    TLWH: typing.ClassVar[BoxFormat]  # value = <BoxFormat.TLWH: 0>
    XYXY: typing.ClassVar[BoxFormat]  # value = <BoxFormat.XYXY: 2>
    __members__: typing.ClassVar[dict[str, BoxFormat]]  # value = {'TLWH': <BoxFormat.TLWH: 0>, 'CXCYWH': <BoxFormat.CXCYWH: 1>, 'XYXY': <BoxFormat.XYXY: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ByteTracker(Tracker):
    def __init__(self, *, low_confidence: typing.SupportsFloat = 0.1, high_confidence: typing.SupportsFloat = 0.6, new_track_confidence: typing.SupportsFloat = 0.7, first_match_cost: typing.SupportsFloat = 0.8, second_match_cost: typing.SupportsFloat = 0.5, tentative_match_cost: typing.SupportsFloat = 0.7, max_lost_frames: typing.SupportsInt = 30, input_format: BoxFormat = ..., threads: typing.SupportsInt = 0) -> None:
        ...
class Detection:
    def __init__(self, x: typing.SupportsFloat, y: typing.SupportsFloat, width: typing.SupportsFloat, height: typing.SupportsFloat, confidence: typing.SupportsFloat, class_id: typing.SupportsInt) -> None:
        ...
    @property
    def class_id(self) -> int:
        ...
    @property
    def confidence(self) -> float:
        ...
    @property
    def height(self) -> float:
        ...
    @property
    def width(self) -> float:
        ...
    @property
    def x(self) -> float:
        ...
    @property
    def y(self) -> float:
        ...
class Prediction:
    @property
    def box(self) -> Detection:
        ...
    @property
    def id(self) -> int:
        ...
class Sort(Tracker):
    def __init__(self, *, max_match_cost: typing.SupportsFloat = 0.7, max_lost_frames: typing.SupportsInt = 30, input_format: BoxFormat = ...) -> None:
        """
        Create SORT. max_match_cost limits 1 - IoU; 0.7 requires IoU >= 0.3.
        """
class Tracker:
    def reset(self) -> None:
        ...
    def update(self, detections: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> numpy.typing.NDArray[numpy.float32]:
        """
        Update one frame: (N, 6) [a, b, c, d, confidence, class_id]. Pixel box coordinates follow input_format: TLWH (x, y, width, height), CXCYWH (center_x, center_y, width, height), or XYXY (x1, y1, x2, y2). Returns float32 (M, 7) [x, y, width, height, track_id, confidence, class_id] with top-left x, y (TLWH), regardless of input_format. Empty frames use shape (0, 6).
        """
    @property
    def input_format(self) -> BoxFormat:
        ...
def add(arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> int:
    """
    A function that adds two numbers
    """
