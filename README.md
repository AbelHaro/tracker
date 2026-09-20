# Tracker C++

Object tracking from per-frame detections. Requires a C++23 compiler, Make,
and Eigen 3 (located at `/usr/include/eigen3` by default).

```sh
make
make run
make test
```

## Python package with uv

The `tracker-cpp` package uses CMake and scikit-build-core to build the Python
extension, imported as `tracker`. It requires Python 3.14 or later, a C++23
compiler, and Eigen 3 headers (`libeigen3-dev` on Debian/Ubuntu). uv installs
the Python build dependencies; the backend obtains CMake and Ninja if needed.

Build a source distribution and wheel from the repository root:

```sh
uv build
```

Build artifacts are written to `dist/`. The backend compiles the C++ sources,
so running `make bindings` first is unnecessary. The wheel targets the platform
and Python version used to build it.

The [uv example](examples/numpy-test/README.md) declares the root package as a
local dependency:

```sh
cd examples/numpy-test
uv sync
uv run numpy-test
```

After changing C++ code, run `uv sync` again from the example directory.
The cache keys include sources and headers so uv rebuilds when they change.
No `PYTHONPATH` configuration is required.

The `src/`, `include/`, `bindings/`, and `tests/` directories remain at the
repository root. The Makefile builds the C++ executable and tests.

## Generate Python type stubs

After changing binding signatures or docstrings, run from the repository root:

```sh
make stubs
```

This builds and installs the current extension in the uv development environment
and generates `bindings/tracker.pyi` with `pybind11-stubgen`. The generator is a
development dependency; it is not needed at runtime. Keep the generated file in
version control and regenerate it instead of editing it manually. CMake installs
it alongside the extension in the wheel.

Run `uv sync` from `examples/numpy-test` afterward to install the updated stub
for IntelliSense. See the [example documentation](examples/numpy-test/README.md#types-and-intellisense)
for editor setup.

## Project structure

- `include/Tracker.hpp`: abstract interface with `update()`, `reset()`, and a virtual destructor.
- `include/ByteTracker.hpp`, `src/ByteTracker.cpp`: configuration and two-stage association.
- `include/Detection.hpp`: bounding box `(x, y, width, height)` in pixels, confidence in `[0, 1]`, class ID, and IoU.
- `include/Prediction.hpp`: output snapshot with a stable ID and estimated box.
- `include/Track.hpp`, `src/Track.cpp`: trajectory, Kalman filter, and `Tentative`, `Tracked`, `Lost`, and `Removed` states.
- `include/KalmanFilter.hpp`: constant-velocity filter with state `[cx, cy, vx, vy]`.
- `include/HungarianAlgorithm.hpp`, `src/HungarianAlgorithm.cpp`: global rectangular assignment with a cost threshold and unmatched rows.
- `bindings/tracker.cpp`: Python bindings and NumPy input/output conversion.
- `bindings/tracker.pyi`: generated Python type declarations for editors and type checkers.
- `src/main.cpp`: example using the abstract interface.
- `tests/`: filter, assignment, track lifecycle, and Python binding tests.

## Usage and extension

```cpp
#include "ByteTracker.hpp"
#include <memory>

std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>();
auto predictions = tracker->update({Detection(10, 20, 40, 60, 0.9)});
for (const auto &prediction : predictions) {
    auto id = prediction.id();
    const auto &box = prediction.box();
    // Use id and box.
}
```

To add another tracker, inherit from `Tracker` and implement `update()` and
`reset()`. The public interface does not depend on Kalman filtering or the
Hungarian algorithm; each implementation can choose its own motion model and
association strategy. `Track` and `HungarianAlgorithm` are reusable components.

Call `update()` once per frame, including frames with no detections. The Kalman
time step is one frame. Results contain only confirmed tracks observed in the
current frame, with Kalman-corrected boxes. Lost tracks remain stored internally
for up to `maxLostFrames` missing frames and can be recovered on the next frame
if they have not exceeded that limit. `reset()` starts a new sequence and resets
IDs to 1. IDs are local to each tracker instance and sequence.

## ByteTrack implementation

1. Predict each track's center with the Kalman filter.
2. Match confirmed and lost tracks to high-confidence detections using
   `1 - IoU * confidence` costs and Hungarian assignment.
3. Match remaining active tracks to low-confidence detections using `1 - IoU`
   costs. Low-confidence detections cannot create or reactivate tracks.
4. Confirm tentative tracks with the remaining high-confidence detections.
   Remove unmatched tentative tracks and create new tracks using `newTrackConfidence`.
5. Retain lost tracks for the configured buffer and remove expired tracks.

Tracks created on the first frame are confirmed immediately; later tracks need
another high-confidence detection on the next frame. Association thresholds are
maximum costs, so lower values are stricter. Assignment first maximizes the
number of valid matches, then minimizes their total cost.

This implementation adapts the two-stage association from
[ByteTrack](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py).
It does not exactly reproduce the reference tracker: it reuses the existing 2D
Kalman filter and retains the last observed width and height instead of also
filtering aspect ratio and height. It does not suppress duplicates between
active and lost tracks. Detections must already be filtered, for example with
NMS. Class IDs are preserved as metadata but do not affect association.

## Eigen optimization

`AssociationCost.hpp` and `src/AssociationCost.cpp` build IoU costs using
coefficient-wise operations on Eigen arrays. Detection coordinates and areas
are prepared once per association, and each track box is retrieved once.
`CostMatrix` uses contiguous row-major storage to match the Hungarian solver's
access pattern. `HungarianAlgorithm::solve()` accepts
`Eigen::Ref<const CostMatrix>` to avoid copying the matrix. Index vectors and
state management still use standard containers.

The Kalman filter uses fixed-size blocks for the observation matrix
`H = [I2, 0]`, LDLT factorization, and the Joseph covariance update. Products
that write to the covariance use separate intermediate values before applying
`noalias()`, following
[Eigen's aliasing rules](https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicAliasing.html).

The Makefile enables `-O2` by default. Override `CXXFLAGS` for debugging.
To compare scalar cost construction with the Eigen implementation:

```sh
make benchmark
```

The benchmark includes matrix preparation and allocation, with 1,000 repetitions
for sizes 16, 64, and 256. It does not measure the complete tracker or guarantee
a particular speedup on other hardware. Tests compare Eigen costs against
scalar IoU and Hungarian assignment against an exhaustively computed optimum.
