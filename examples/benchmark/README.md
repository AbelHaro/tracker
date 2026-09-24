# Tracker benchmark

Compare the C++ tracker with Ultralytics ByteTrack using detections from the
same video and YOLO model as the `examples/yolo` application:

```sh
uv run --package benchmark benchmark
```

The benchmark reads the video and runs YOLO once, then replays the cached
detections through both trackers. Video decoding and model inference are
excluded from timing. It does not display or save video, wait between frames,
or print per-frame results, so processing runs as fast as the model and machine
allow. Only calls to each tracker's `update` method are timed.

Choose another video or model, or repeat the tracker pass for steadier timing:

```sh
uv run --package benchmark benchmark --video path/to/video.mp4 --model path/to/yolo.pt --repeats 5
```

It uses the same detection confidence threshold and tracker configuration as
the YOLO example. Average tracks per frame are reported to help compare results.
