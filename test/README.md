# Testing

Run these commands from the repository root. For linting and development
setup, see [Development](../documentation/development.md).

## Run the core tests

```sh
python3 scripts/run-tests.py --core-only
```

These test search and smoothing with ordinary Python, without QGIS, GDAL, or
NumPy. The full suite below also includes them.

## Run the full suite with Docker

Build a ZIP with a fixed filename for testing:

```sh
python3 scripts/package-plugin.py --output dist/test-plugin.zip
```

Run it in QGIS 4:

```sh
docker run --rm --network none \
  -e QT_QPA_PLATFORM=offscreen -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --workdir /workspace --entrypoint python3 \
  qgis/qgis:4.0-trixie \
  scripts/run-tests.py --plugin-zip dist/test-plugin.zip \
  --expected-qgis 4.0 --expected-qt-major 6
```

Repeat using each row below:

| Image | `--expected-qgis` | `--expected-qt-major` |
| --- | --- | --- |
| `qgis/qgis:3.40-bookworm` | `3.40` | `5` |
| `qgis/qgis:3.44-bookworm` | `3.44` | `5` |
| `qgis/qgis:4.0-trixie` | `4.0` | `6` |

Docker must be installed and accessible to your user. It may download the
image on the first run. To reproduce CI's exact environments, use the image
digests in the [workflow](../.github/workflows/compatibility.yml).

Tests load the plugin from the ZIP and exercise real QGIS providers, widgets,
editing, and background tasks. Test data and settings use temporary locations.
The repository is mounted read-only; the test container has no network access.

## Run with a local QGIS installation

Use a Python interpreter configured for PyQGIS:

```sh
make test PYTHON=/path/to/qgis/python
```

This tests the source tree. Do not use `uv run make test`: uv's environment
contains development tools, not QGIS's runtime.

## Read the results

The full runner prints the QGIS, Qt, PyQt, and NumPy versions. A mismatch with
an expected QGIS or Qt version fails the run. Tests return a nonzero exit code
on failure and the runner stops after 180 seconds if it hangs.

The translation test uses the separate `test/af.qm` fixture. If it is missing,
that test skips; report the skip alongside the test results. Rebuild it with:

```sh
lrelease test/af.ts -qm test/af.qm
```

This fixture checks Qt translation loading, not the completeness of the
plugin's translations.

## Check in QGIS

Use a clean profile for QGIS 3 and QGIS 4. Install the ZIP and copy the
[example project](../documentation/example/) to a writable location.
These checks cover desktop behavior that the automated suite cannot establish.

| Action | Expected result |
| --- | --- |
| Follow [Trace your first line](../documentation/usage.md#trace-your-first-line), including enabling editing | The line follows the raster, matches the preview, and stays after finishing. |
| Use `A`, `D`, `T`, and `S`; repeat with Layers-panel focus during a trace | Modes, sampled color, and color snapping change as described in the usage guide. |
| Press Escape during a long trace; start another | Earlier segments remain and the new trace can finish. |
| Press `B` during pending work, then after a completed segment | Pending work cancels; the last tracing edit can be undone. |
| Enable both snapping options and smoothing | The endpoint joins the intended vertex; preview and saved geometry agree. |
| Change the raster, vector layer, or CRS during a trace | Earlier geometry remains and a late result does not add another segment. |
| Close and reopen the dock during a trace; unload while work is pending | The dock and tool close without a crash or stray result. |
| Change preview color and width; reopen the dock | The saved appearance is restored. |
| Save edits and reopen the project | The saved line remains in the vector layer. |
| Trace a representative large raster | Hover feedback, cancellation, and panning remain usable. |
| On QGIS 3, load Raster Tracer too and switch between the tools | Their names, icons, settings, and shortcuts remain separate; unloading either leaves the other usable. |

Record the tested commit or working-tree changes, QGIS version, platform,
and any failures. Keep dated test reports separate from this guide.
