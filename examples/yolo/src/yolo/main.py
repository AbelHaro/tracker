from pathlib import Path

import cv2
import numpy as np
import tracker
from ultralytics import YOLO

VIDEO_URL = str(Path(__file__).with_name("traffic.mp4"))


def main() -> None:
    model = YOLO("yolo26n.pt", task="detect")

    tracker_config = tracker.ByteTrackerConfig()
    tracker_config.inputFormat = tracker.BoxFormat.CXCYWH

    byte_tracker = tracker.ByteTracker(tracker_config)

    video = cv2.VideoCapture(VIDEO_URL)

    if not video.isOpened():
        print("Error opening video stream or file")
        return

    for frame_num in range(max(1_000, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25, verbose=False)

        boxes = results[0].boxes
        detections = np.empty((len(boxes), 6), dtype=np.float32)

        detections[:, :4] = boxes.xywh.cpu().numpy()
        detections[:, 4] = boxes.conf.cpu().numpy()
        detections[:, 5] = boxes.cls.cpu().numpy()

        tracks = byte_tracker.update(detections)

        print(f"Frame {frame_num}: {len(tracks)} tracks")
        print(f"Tracks: {tracks}")

        for track in tracks:
            x = track[0]
            y = track[1]
            w = track[2]
            h = track[3]
            track_id = track[4]
            confidence = track[5]
            class_id = track[6]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (int(x), int(y) - 10),
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
