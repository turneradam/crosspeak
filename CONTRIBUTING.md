# Contributing to crosspeak

Contributions are welcome — bug reports, feature requests, documentation, and code.

## Getting help

Open an [Issue](https://github.com/turneradam/crosspeak/issues) for bugs and
feature requests.

## Reporting a bug

Please include:
- crosspeak version (`python -c "import crosspeak; print(crosspeak.__version__)"`)
- Python version and OS
- A minimal example that reproduces the problem
- What you expected, and what happened instead

## Development setup

crosspeak uses [uv](https://docs.astral.sh/uv/) for dependency management and
[just](https://github.com/casey/just) as a task runner.

```bash
git clone https://github.com/turneradam/crosspeak.git
cd crosspeak
uv sync
just test      # pytest
just lint      # ruff check
just format    # ruff format
```

Python 3.11 or newer is required.

## Pull requests

1. Open an issue first for anything larger than a typo, so the approach can be
   agreed before you spend time on it.
2. Branch from `main`.
3. Add tests. New numerical code needs a test that would fail without it.
4. Ensure `just test` and `just lint` pass.
5. Anything exported from `crosspeak/__init__.py` needs a docstring.

## Scope

crosspeak implements generalized 2D correlation spectroscopy for vibrational
spectra. Contributions extending the correlation formalism (codistribution,
sample–sample correlation, higher-order analysis) are in scope. General-purpose
spectral processing well served by scipy or scikit-learn is not.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
