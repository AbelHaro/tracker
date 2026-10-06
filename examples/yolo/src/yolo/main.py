from pathlib import Path
from typing import cast

import cv2
import numpy as np
import tracker
from ultralytics import YOLO
from ultralytics.engine.results import Results

VIDEO_URL = str(Path(__file__).with_name("traffic.mp4"))


def main() -> None:
    model = YOLO("yolo26n.pt", task="detect")

    byte_tracker = tracker.ByteTracker(input_format=tracker.BoxFormat.CXCYWH)

    video = cv2.VideoCapture(VIDEO_URL)

    if not video.isOpened():
        print("Error opening video stream or file")
        return

    for frame_num in range(max(1_000, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = video.read()
        if not ret:
            break

        # Non-streaming detection returns a list of Results.
        results = cast(
            list[Results], model.predict(frame, conf=0.25, verbose=False, stream=False)
        )

        boxes = results[0].boxes
        if boxes is None:
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            boxes = boxes.cpu().numpy()
            detections = np.empty((len(boxes), 6), dtype=np.float32)
            detections[:, :4] = np.asarray(boxes.xywh)
            detections[:, 4] = np.asarray(boxes.conf)
            detections[:, 5] = np.asarray(boxes.cls)

        tracks = byte_tracker.update(detections)

        print(f"Frame {frame_num}: {len(tracks)} tracks")
        print(f"Tracks: {tracks}")

        for track in tracks:
            box = track.box
            cv2.rectangle(
                frame,
                (int(box.x), int(box.y)),
                (int(box.x + box.width), int(box.y + box.height)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"ID: {track.id}",
                (int(box.x), int(box.y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YOLO26n + ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
