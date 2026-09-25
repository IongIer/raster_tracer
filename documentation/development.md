# Development

Run commands from the repository root. QGIS provides Qt, GDAL, and NumPy at
runtime. The separate uv environment contains development tools only.

## Build the plugin

```sh
python3 scripts/package-plugin.py
```

This creates `dist/raster_scribe-<version>.zip` from the working tree, including
uncommitted changes. It needs ordinary Python 3, not a QGIS installation.
Install the ZIP through QGIS's plugin manager.

Use `make package` for the same build. When adding runtime files, include them
in the package file list in `pb_tool.cfg`.

## Check Python changes

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.8.4 or
newer and use Python 3.9+ for the tooling environment:

```sh
uv sync --locked
make lint
make format-check
```

`make format` applies formatting. To apply lint fixes, run
`uv run --locked ruff check --fix .` and review the diff.

The [testing guide](../test/README.md) covers core tests, the packaged QGIS
matrix, and desktop checks. Do not run `uv run make test` or install QGIS
runtime packages in `.venv`; use QGIS's Python or the containers.

The [CI workflow](../.github/workflows/compatibility.yml) runs lint, formatting,
and packaged QGIS tests on pushes to `master` and manual runs.

## Update icons

After editing `icon.png` or `resources.qrc`, regenerate the resource module:

```sh
make compile
```

This requires `pyrcc5`. Include the regenerated `resources.py` with the changes.

## Add or update a translation

To add or update a translation, use `pylupdate5` with its language code
(shown here as `de`):

```sh
scripts/update-strings.sh de
```

Translate the entries in `i18n/de.ts`, then compile with Qt's `lrelease`:

```sh
scripts/compile-strings.sh lrelease de
```

Include both the `.ts` and `.qm` files when adding or updating a translation.

## Update the documentation and example

Edit the Markdown files under `documentation/`. There is no documentation
build step.

The [example project](example/) contains a generated RGB raster and an empty
MultiLineString GeoPackage. To recreate it with a configured PyQGIS Python:

```sh
python3 scripts/create-example.py
```

This replaces the example files, including any traced geometry saved there.
Copy the example before using it for manual tracing.

## Profile tracing

To collect timings, start QGIS with:

```sh
RASTER_SCRIBE_PROFILE=1 qgis
```

Read the **Raster Scribe** tab in the Log Messages panel. It includes raster
sampling, search, and postprocessing timings and a cProfile summary. Restart
without the variable for normal use.
