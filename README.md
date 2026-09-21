# Tracker C++

Object tracking library under development. It receives detections from each
frame and maintains an ID for each object through a C++ ByteTrack implementation
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

The tracker output has the following format:

```text
[x, y, width, height, id, confidence, class]
```

Coordinates are given in pixels, and the example uses the `CXCYWH` format,
which is compatible with Ultralytics `boxes.xywh`.

## C++ executable and tests

To build the example executable and run the native tests:

```sh
make
make run
make test
```
