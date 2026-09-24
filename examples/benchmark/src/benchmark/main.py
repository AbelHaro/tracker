"""Compare tracker update times on detections from the YOLO example video."""

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from shutil import get_terminal_size
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from tracker import BoxFormat, ByteTracker, ByteTrackerConfig
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

DETECTION_CONFIDENCE = 0.25


class ProgressLine:
    """Refresh one terminal line at most four times per second."""

    def __init__(self):
        self.last_refresh = 0.0
        self.width = 0

    def update(self, message: str, *, force: bool = False) -> None:
        now = perf_counter()
        if not force and now - self.last_refresh < 0.25:
            return
        columns = max(1, get_terminal_size().columns - 1)
        message = message[:columns]
        self.width = min(columns, max(self.width, len(message)))
        print("\r" + message.ljust(self.width), end="", file=sys.stderr, flush=True)
        self.last_refresh = now

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("\r" + " " * self.width + "\r", end="", file=sys.stderr, flush=True)


def collect_detections(
    model: YOLO, video_path: Path, progress: ProgressLine
) -> tuple[list[np.ndarray], list[Boxes]]:
    """Run YOLO once and cache both trackers' inputs for timing-only replays."""
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_frames = []
    ultralytics_frames = []
    total_frames = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    started = perf_counter()
    progress.update("YOLO: preparing video detections...", force=True)
    try:
        while True:
            read_ok, frame = video.read()
            if not read_ok:
                break

            results = model.predict(
                frame, conf=DETECTION_CONFIDENCE, verbose=False, show=False
            )
            boxes = results[0].boxes.cpu()
            xywh = boxes.xywh.numpy()
            confidence = boxes.conf.numpy()
            class_id = boxes.cls.numpy()

            detections = np.empty((len(boxes), 6), dtype=np.float32)
            detections[:, :4] = xywh
            detections[:, 4] = confidence
            detections[:, 5] = class_id
            native_frames.append(detections)
            ultralytics_frames.append(boxes)
            processed = len(native_frames)
            total = str(total_frames) if total_frames else "?"
            fps = processed / (perf_counter() - started)
            progress.update(
                f"YOLO {processed}/{total} | {len(boxes)} detections | {fps:.1f} fps"
            )
    finally:
        video.release()

    if not native_frames:
        raise RuntimeError(f"The video contains no readable frames: {video_path}")
    return native_frames, ultralytics_frames


def make_native_tracker() -> ByteTracker:
    config = ByteTrackerConfig()
    config.inputFormat = BoxFormat.CXCYWH
    return ByteTracker(config)


def make_ultralytics_tracker() -> BYTETracker:
    return BYTETracker(
        Namespace(
            tracker_type="bytetrack",
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.7,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
    )


def measure_updates(
    tracker, frames, progress: ProgressLine, label: str
) -> tuple[float, float]:
    """Return summed update-call time and average output tracks per frame."""
    update_seconds = 0.0
    track_count = 0
    progress.update(f"{label} | 0/{len(frames)} frames", force=True)
    for frame_index, detections in enumerate(frames, start=1):
        start = perf_counter()
        tracks = tracker.update(detections)
        update_seconds += perf_counter() - start
        track_count += len(tracks)
        progress.update(
            f"{label} | {frame_index}/{len(frames)} | {len(tracks)} tracks"
            f" | {update_seconds * 1000 / frame_index:.3f} ms/update"
        )
    return update_seconds, track_count / len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    video_path = Path(__file__).parent.parent / "benchmark" / "traffic.mp4"

    with ProgressLine() as progress:
        progress.update("Loading YOLO model...", force=True)
        model = YOLO("yolo26n", task="detect")
        native_frames, ultralytics_frames = collect_detections(
            model, video_path, progress
        )

        # Warm each implementation on a separate tracker instance.
        progress.update("Warming up trackers...", force=True)
        make_native_tracker().update(native_frames[0])
        make_ultralytics_tracker().update(ultralytics_frames[0])

        runners = (
            ("C++ tracker", make_native_tracker, native_frames),
            ("Ultralytics", make_ultralytics_tracker, ultralytics_frames),
        )
        samples: dict[str, list[float]] = {name: [] for name, _, _ in runners}
        mean_tracks: dict[str, float] = {}
        for repeat in range(args.repeats):
            ordered_runners = runners if repeat % 2 == 0 else tuple(reversed(runners))
            for name, factory, frames in ordered_runners:
                tracker = factory()
                label = f"{name} run {repeat + 1}/{args.repeats}"
                elapsed, mean_tracks[name] = measure_updates(
                    tracker, frames, progress, label
                )
                samples[name].append(elapsed)

    frame_count = len(native_frames)
    print(f"Video: {video_path}; {frame_count} frames")
    print(f"Frames: {frame_count}; YOLO confidence threshold: {DETECTION_CONFIDENCE}")
    print(f"Tracker runs: {args.repeats}; inference and video decoding are excluded")
    print("tracker       median_update_s   ms/update   mean tracks/frame")
    for name, times in samples.items():
        elapsed = median(times)
        print(
            f"{name:<13} {elapsed:>16.4f} {elapsed * 1000 / frame_count:>11.3f}"
            f" {mean_tracks[name]:>18.1f}"
        )


if __name__ == "__main__":
    main()
