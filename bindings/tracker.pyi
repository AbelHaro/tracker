"""
Tracker library
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = [
    "ByteTracker",
    "ByteTrackerConfig",
    "Detection",
    "Prediction",
    "Tracker",
    "add",
]

class ByteTracker(Tracker):
    def __init__(self, config: ByteTrackerConfig = ...) -> None: ...

class ByteTrackerConfig:
    def __init__(self) -> None: ...
    @property
    def firstMatchCost(self) -> float: ...
    @firstMatchCost.setter
    def firstMatchCost(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def highConfidence(self) -> float: ...
    @highConfidence.setter
    def highConfidence(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def lowConfidence(self) -> float: ...
    @lowConfidence.setter
    def lowConfidence(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def maxLostFrames(self) -> int: ...
    @maxLostFrames.setter
    def maxLostFrames(
        self, arg0: typing.SupportsInt | typing.SupportsIndex
    ) -> None: ...
    @property
    def newTrackConfidence(self) -> float: ...
    @newTrackConfidence.setter
    def newTrackConfidence(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def secondMatchCost(self) -> float: ...
    @secondMatchCost.setter
    def secondMatchCost(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def tentativeMatchCost(self) -> float: ...
    @tentativeMatchCost.setter
    def tentativeMatchCost(
        self, arg0: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...

class Detection:
    def __init__(
        self,
        x: typing.SupportsFloat | typing.SupportsIndex,
        y: typing.SupportsFloat | typing.SupportsIndex,
        width: typing.SupportsFloat | typing.SupportsIndex,
        height: typing.SupportsFloat | typing.SupportsIndex,
        condifence: typing.SupportsFloat | typing.SupportsIndex,
        class_id: typing.SupportsInt | typing.SupportsIndex,
    ) -> None: ...
    @property
    def class_id(self) -> int: ...
    @property
    def confince(self) -> float: ...
    @property
    def height(self) -> float: ...
    @property
    def width(self) -> float: ...
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...

class Prediction:
    @property
    def box(self) -> Detection: ...
    @property
    def id(self) -> int: ...

class Tracker:
    def reset(self) -> None: ...
    def update(
        self, detections: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
    ) -> numpy.typing.NDArray[numpy.float32]:
        """
        Update one frame: (N, 6) [x, y, width, height, confidence, class_id] to float32 (N, 7) [x, y, width, height, track_id, confidence, class_id], where N is the number of returned tracks.
        """

def add(
    arg0: typing.SupportsInt | typing.SupportsIndex,
    arg1: typing.SupportsInt | typing.SupportsIndex,
) -> int:
    """
    A function that adds two numbers
    """
