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

The builder uses the file list in `pb_tool.cfg`, includes the existing
`LICENSE` and compiled translations, and fails if a listed file is missing.
Tests, example data, working notes, and development dependencies are excluded.
The `pb_tool` application is not required.

Use `make package` for the same build.

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

Ruff is pinned in `pyproject.toml` and `uv.lock`. Upgrade it deliberately and
update both files together. Generated `resources.py` is excluded from Ruff.

The [testing guide](../test/README.md) covers core tests, the packaged QGIS
matrix, and desktop checks. Do not run `uv run make test` or install QGIS
runtime packages in `.venv`; use QGIS's Python or the containers.

## Icons and translations

After editing `icon.png` or `resources.qrc`, regenerate the resource module:

```sh
make compile
```

This requires `pyrcc5`. The build rewrites its output to import Qt through
`qgis.PyQt`, so the same checked-in `resources.py` works with Qt5 and Qt6.
Include the regenerated module when sharing resource changes.

To add or update a translation, use `pylupdate5` with its language code
(shown here as `de`):

```sh
scripts/update-strings.sh de
```

Translate the entries in `i18n/de.ts`, then compile with Qt's `lrelease`:

```sh
scripts/compile-strings.sh lrelease de
```

The collector reads runtime Python and UI files from `pb_tool.cfg`. It does
not scan development environments or working notes. Runtime catalogs are
named `i18n/<language>.qm` and use Raster Scribe's translation contexts.
Include both the `.ts` and `.qm` files when adding or updating a translation.
The separate `test/af.qm` fixture is described in the testing guide.

## Documentation and example

Public documentation is Markdown under `documentation/`. The ignored `docs/`
directory is reserved for local working notes. There is no documentation
build step; `make doc` prints the location of the Markdown files.

The [example project](example/) contains a generated RGB raster and an empty
MultiLineString GeoPackage. To recreate it with a configured PyQGIS Python:

```sh
python3 scripts/create-example.py
```

This replaces the example files, including any traced geometry saved there.
Copy the example before using it for manual tracing.

## Profiling and tracing internals

To collect timings, start QGIS with:

```sh
RASTER_SCRIBE_PROFILE=1 qgis
```

Read the **Raster Scribe** tab in the Log Messages panel. It includes raster
sampling, search, and postprocessing timings and a cProfile summary. Restart
without the variable for normal use.

The search moves through four neighboring pixels and minimizes the sum of
entered-pixel squared RGB distances, quantized to integers. Zero-cost pixels
remain zero cost. It searches a deterministic window with 1024 pixels of
padding and a minimum dimension of 512, clipped to the raster. Cache history
does not change the searched area.

Caps are 8,388,608 search pixels, 256 MiB of accounted array and scratch
allocations, 1,000,000 discovered nodes, and 1,000,000 frontier entries. The
array cap includes a reserve for cursor sampling. These limits do not bound
total QGIS, GDAL, or Python memory use.

Large reads, cost preparation, search, and smoothing run in a background task.
One scheduler survives dock close/reopen so cancelled workers drain before
replacement work starts. Python search can still contend for the GIL.

With smoothing enabled, a moving average is followed by QGIS's line simplifier
with a distance tolerance of 0.25 raster pixels. Both operations preserve the
segment endpoints, and previews and saved paths use the same result.

Extending a 2D multipart line uses QGIS's native geometry append. Earlier
geometries remain available to QGIS undo, so memory can grow substantially when
many segments extend one long feature. Finishing a line releases the tracing
cache and markers but preserves the layer's undo history.

## Continuous integration

The [workflow](../.github/workflows/compatibility.yml) runs lint and formatting
checks, then builds and tests the ZIP in QGIS 3.40, 3.44, and 4.0 containers.
It runs on pushes to `master` and manual dispatches. It does not publish the
plugin. Container digests are pinned there; update them when testing newer
QGIS builds.
