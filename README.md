# Image Processing

This repository is an early-stage image processing project written in Python.

It is still under active restructuring, so the current focus is on composing and organizing complete implementations rather than providing a finished public package or formal documentation.

The goal of this project is not to replace mature image processing libraries. The code is intentionally written in pure Python, which makes some optimizations harder and can limit execution speed compared with low-level implementations. Even so, the package aims to provide complete, practical techniques that can be used quickly in experiments and small workflows.

Where appropriate, the project uses `numba` to improve execution times in performance-sensitive routines.

## Project Status

- Early development version.
- Internal structure is still being reorganized.
- Package-level documentation is not available yet.
- Examples are still the main place to explore features and experiment with implementations.

## Repository Layout

- `pixelops/`: core library code grouped by topic.
- `examples/`: example implementations and usage experiments.
- `tests/`: automated tests when available.
- `data/`: input assets used by examples and experiments.
- `benchmarks/`: scripts for measuring performance of selected algorithms.

## Examples

The `examples/` folder contains implementation demos and work-in-progress scripts.

These examples are useful for:

- exploring how an algorithm is implemented,
- testing visual output,
- comparing variants of the same idea,
- iterating before code is promoted into the library.

## Package

The library is being built incrementally and is not yet fully documented.

If you are using the package directly, inspect the source code inside `pixelops/` to understand the available functions and expected inputs.

## Requirements

The project uses Python together with common scientific and image processing packages such as:

- OpenCV
- NumPy
- Matplotlib
- SciPy
- Numba

Some modules may also depend on optional packages for performance or specific algorithms.

## Running Examples

Most example scripts are designed to be executed directly with Python.

Typical workflow:

1. Install the required dependencies.
2. Run the example script from the project root.
3. Use the images in `data/` or adjust the input path in the script.

## Development Notes

- This project is being refactored in small steps.
- Some files are experimental or temporary.
- Naming and folder organization may change as the library matures.
- Documentation will be expanded once the core structure stabilizes.

## Contributing

If you are modifying the project, keep changes small and consistent with the existing style.

When possible:

- prefer readable implementations over premature optimization,
- keep examples self-contained,
- move stable code from `examples/` into `pixelops/` only when it is ready for reuse,
- add tests alongside new reusable functionality.
