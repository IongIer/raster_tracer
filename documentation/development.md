# Development

Run commands from the repository root. QGIS supplies Qt, GDAL, and NumPy;
the uv environment contains development tools only.

## Build

```sh
python3 scripts/package-plugin.py
```

This creates `dist/raster_scribe-<version>.zip` from the working tree, including
uncommitted changes, using ordinary Python 3. `make package` does the same.
Add new runtime files to `pb_tool.cfg`.

## Check changes

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.8.4+ and
Python 3.9+:

```sh
uv sync --locked
make lint
make format-check
```

`make format` applies formatting; `uv run --locked ruff check --fix .` applies
lint fixes. Review the changes before committing.

Follow the [testing guide](../test/README.md) for local tests and the packaged
QGIS matrix. Do not use `uv run make test` or install QGIS
runtime packages in `.venv`. The [CI workflow](../.github/workflows/compatibility.yml)
runs lint, formatting, and packaged tests on `master` pushes and manual runs.

## Update resources and translations

After editing icons or `resources.qrc`, run `make compile` with `pyrcc5`
available and include the regenerated `resources.py`.

To update a language, extract strings with `pylupdate5` available:

```sh
scripts/update-strings.sh de
```

Edit `i18n/de.ts`, then compile with Qt's `lrelease`:

```sh
scripts/compile-strings.sh lrelease de
```

Include both `.ts` and `.qm` files.

## Profile tracing

Start QGIS with `RASTER_SCRIBE_PROFILE=1 qgis`. The **Raster Scribe** tab in
Log Messages shows timings and a cProfile summary. Restart without the
variable for normal use.
