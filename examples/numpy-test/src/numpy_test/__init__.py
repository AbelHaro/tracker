import numpy as np
import tracker
from numpy.typing import NDArray


def main() -> None:
    tracker_config = tracker.ByteTrackerConfig()
    byte_tracker = tracker.ByteTracker(tracker_config)

    # x, y, width, height, confidence, class_id
    detections: NDArray[np.float32] = np.array(
        [
            [100, 100, 50, 50, 0.9, 1],
            [200, 200, 60, 60, 0.8, 2],
            [300, 300, 70, 70, 0.9, 1],
        ],
        dtype=np.float32,
    )
    tracks = byte_tracker.update(detections)
    print("x, y, width, height, track_id, confidence, class_id")
    print(tracks)

    for i in range(100):
        tracks = byte_tracker.update(
            np.array(
                [
                    [100 + i, 100 + i, 50, 50, 0.9, 1],
                    [200 + i, 200 + i, 60, 60, 0.8, 2],
                    [300 + i, 300 + i, 70, 70, 0.9, 1],
                ],
                dtype=np.float32,
            )
        )

        if i % 10 == 0:
            print(f"Frame {i}:")
            print("x, y, width, height, track_id, confidence, class_id")
            print(tracks)
