# uv example

This project uses NumPy and the root C++ package (`tracker-cpp`). Its local
dependency is declared in `pyproject.toml` with `path = "../.."`.

Requires Python 3.14 or later, a C++23 compiler, and Eigen 3. From this directory:

```sh
uv sync
uv run python -c "import tracker; print(tracker.add(2, 3))"
uv run main
```

uv builds and installs the binding automatically. The distribution is named
`tracker-cpp`; the importable module is named `tracker`.

The entry point is `src.main:main`, defined in `src/main.py`.
There is no `__init__.py`; `namespace = true` enables packaging this layout.
The command `uv run numpy-test` remains available as an alias.

## NumPy interface

`ByteTracker.update(detections)` accepts a NumPy array with shape `(N, 6)`:
`[x, y, width, height, confidence, class_id]`. Coordinates describe the top-left
corner and size of the bounding box in pixels. `class_id` must be a finite
integer representable as a C++ `int`.

The result is a `float32` array with shape `(N, 7)` and columns
`[x, y, width, height, track_id, confidence, class_id]`. Here, `N` is the number
of returned tracks, which may differ from the number of input detections.
`class_id` comes from the last matched detection and does not affect association.

Noncontiguous arrays and arrays with other numeric dtypes are converted to
contiguous `float32` arrays automatically. For a frame without detections, pass
`np.empty((0, 6), dtype=np.float32)`. An empty result has shape `(0, 7)`.
IDs are also returned as `float32`; integers above `2**24` may lose precision.

Run the binding tests from this directory:

```sh
uv run python ../../tests/test_numpy_binding.py
```

## Types and IntelliSense

The package installs `tracker.pyi` alongside the compiled extension. This file
declares classes, properties, and methods so Pylance can provide completion and
infer the return type of `update()` as `NDArray[np.float32]`. The input shape
`(N, 6)` and output shape `(M, 7)` are documented on the method; `NDArray` does
not statically check the number of columns.

In VS Code, with the Python and Pylance extensions installed:

1. Run `uv sync` from this directory.
2. Open `Ctrl+Shift+P` and select `Python: Select Interpreter`.
3. Select `examples/numpy-test/.venv/bin/python` from the repository root,
   or `.venv/bin/python` if you opened only the example directory.

Typing `tracker.`, `tracker_config.`, or `byte_tracker.` shows their members.
Hovering over `tracks` shows its inferred array type. If the editor retains
outdated information, run `Python: Restart Language Server`.

When the C++ API changes, regenerate the stub from the repository root:

```sh
make stubs
cd examples/numpy-test
uv sync
```

`make stubs` uses the uv development environment to build and install the current
extension, then runs `pybind11-stubgen --exit-code -o bindings tracker`. The
`--exit-code` option makes generation fail if the generator reports errors.
The generated `bindings/tracker.pyi` is included in the wheel. Do not edit it
manually; change signatures and docstrings in `bindings/tracker.cpp` instead.

### Why use a .pyi file?

[Type stubs are the standard way to describe compiled extension modules](https://typing.python.org/en/latest/spec/distributing.html#stub-files).
They provide static type information without changing runtime behavior.
This project uses [pybind11-stubgen](https://github.com/pybind/pybind11-stubgen)
to generate the file from the built module. Review generated changes when the
API changes, especially array types and default arguments. Array shapes and
semantic constraints remain documented in the C++ binding docstrings and
validated at runtime.
