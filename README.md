# Tracker C++

Object tracking library under development. It receives detections from each
frame and maintains an ID for each object through C++ ByteTrack and SORT implementations
with Python bindings.

## Requirements

- Linux, Python 3.14 or later, and [uv](https://docs.astral.sh/uv/).
- A C++23-compatible compiler.
- CMake 3.20 or later.
- Eigen 3 and its development headers.

On Debian or Ubuntu:

```sh
sudo apt install build-essential cmake libeigen3-dev python3.14-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Build

From the repository root, configure and build the Python extension:

```sh
cmake -S . -B build
cmake --build build
```

To install the Python dependencies and build the `tracker` package with uv:

```sh
uv sync
```

The package is imported from Python as `tracker`. You can also build a wheel
with:

```sh
uv build
```

## Python interface

Create either implementation with keyword-only arguments; configuration classes
have been removed:

```python
from tracker import BoxFormat, ByteTracker, Sort, Tracker

sort = Sort(max_match_cost=0.7, max_lost_frames=30, input_format=BoxFormat.XYXY)
byte = ByteTracker(high_confidence=0.6, input_format=BoxFormat.XYXY, threads=0)

# Both implement Tracker.update(), reset(), and the read-only input_format property.
model: Tracker = sort
tracks = model.update(detections)
model.reset()
```

`update` accepts NumPy `float32` arrays of shape `(N, 6)` with box coordinates,
confidence, and class ID. Sliced and transposed arrays are supported. Convert
other inputs with `np.asarray(detections, dtype=np.float32)` before calling.
It returns a `list[Prediction]` with integer track IDs and detection boxes,
always in TLWH format.
Use `(0, 6)` arrays for empty frames. Python names use snake_case;
`Detection` exposes `confidence`.

Results are C++ output snapshots with named properties:

```python
for prediction in model.update(detections):
    box = prediction.box
    print(prediction.id, box.x, box.y, box.width, box.height,
          box.confidence, box.class_id)
```

Call `update` once per frame to advance the tracker.
Snapshots remain valid after subsequent updates or resets.

SORT uses one Hungarian association pass without confidence filtering or fusion.
`max_match_cost` limits **1 - IoU**: 0.7 requires IoU of at least 0.3.
First-frame births are returned immediately; later births need a match in the
next frame. Lost tracks retain their IDs for up to `max_lost_frames` missing
frames and are returned only when matched.

ByteTracker additionally accepts `low_confidence`, `high_confidence`,
`new_track_confidence`, `first_match_cost`, `second_match_cost`,
`tentative_match_cost`, and `threads` (0 selects automatically).
C++ implementations also take constructor arguments directly, with defaults.

Bindings are organized into shared types (`common.cpp`), NumPy conversion
(`arrays.cpp`), and one registration file per implementation. To expose another
tracker, register its constructor as a subclass of `Tracker`, call its binding
function from `bindings/tracker.cpp`, and add its source to CMake.

## Examples workspace

All projects under `examples/*` belong to the uv workspace and share the root
`.venv` and `uv.lock`. Install all examples and run the NumPy plot with:

```sh
uv sync --all-packages
uv run --all-packages tracking-demo
```

Select the root `.venv/bin/python` in your editor for Python and ty. Use
`--all-packages` when running examples to keep all their dependencies installed.

## Run the YOLO example

The example uses YOLO and OpenCV to detect objects in a video and pass them to
the tracker. From the repository root:

```sh
uv sync --all-packages
uv run --all-packages traffic-tracker
```

You can also run it from `examples/yolo`:

```sh
cd examples/yolo
uv sync --all-packages
uv run --all-packages traffic-tracker
```

The example uses `examples/yolo/src/yolo/traffic.mp4` and downloads the
`yolo26n.pt` weights on the first run. Press `Q` to close the window.

The example calls `ByteTracker.update` with a float32 NumPy array of
shape `(N, 6)` and receives `Prediction` objects. Each exposes `id` and `box`;
the box exposes `x`, `y`, `width`, `height`, `confidence`, and `class_id`.

Coordinates are given in pixels, and the example uses the `CXCYWH` format,
which is compatible with Ultralytics `boxes.xywh`.

## C++ executable and tests

To build the example executable, run it, and run the native tests:

```sh
cmake -S . -B build
cmake --build build
cmake --build build --target run
ctest --test-dir build --output-on-failure
```

Run the association benchmark or regenerate Python stubs with:

```sh
cmake --build build --target benchmark
cmake --build build --target stubs
```

The `stubs` target requires `uv`. Use `cmake --build build --target clean`
to remove CMake build outputs.
