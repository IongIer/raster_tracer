# Compatibility validation

The test suite uses real QGIS providers, Qt widgets, map canvases and background
tasks. `qgis_interface.py` supplies the small part of `QgsInterface` the plugin
needs; it is not the complete QGIS desktop application.

The 0.3.4 ZIP was validated with 15 passing tests and one optional translation
fixture skipped in each of these environments:

| QGIS | Qt / PyQt | NumPy |
| --- | --- | --- |
| 3.40.12 | 5.15.8 / 5.15.9 | 1.24.2 |
| 3.44.4 | 5.15.8 / 5.15.9 | 1.24.2 |
| 4.0.3 | 6.8.2 / 6.9.0 | 2.2.4 |

Tests cover plugin startup and reopening, icon loading, keyboard shortcuts,
mouse clicks, color snapping and settings, tracing and committing geometry,
background trace/preview cancellation and restart, and differing
project/raster/vector CRSs. A
16-by-16 RGB GeoTIFF is generated in a temporary directory and deleted after
each integration test. This also checks conversion from GDAL byte bands to
floating-point arrays under NumPy 1 and 2.

## Run without installing QGIS on the host

Build the ZIP with the host's Python (no QGIS needed):

```sh
python3 scripts/package-plugin.py
```

Run the suite against that exact ZIP in QGIS 4:

```sh
docker run --rm --network none \
  -e QT_QPA_PLATFORM=offscreen -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --workdir /workspace --entrypoint python3 \
  qgis/qgis:4.0-trixie \
  scripts/run-tests.py --plugin-zip dist/raster_tracer-0.3.4.zip \
  --expected-qgis 4.0 --expected-qt-major 6
```

Repeat with the following image and version arguments:

| Image | `--expected-qgis` | `--expected-qt-major` |
| --- | --- | --- |
| `qgis/qgis:3.40-bookworm` | `3.40` | `5` |
| `qgis/qgis:3.44-bookworm` | `3.44` | `5` |
| `qgis/qgis:4.0-trixie` | `4.0` | `6` |

The [CI workflow](../.github/workflows/compatibility.yml) pins these images by
digest for reproducibility. Use its `tag@sha256:...` references to reproduce
the exact environments. The runner prints QGIS, Qt, PyQt and NumPy versions
and fails if the requested QGIS/Qt versions do not match. Update image digests
deliberately when testing newer patch releases.

The plugin ZIP is extracted into a temporary directory. Tests import code and
resources from that directory, not the source checkout. Settings also use
temporary locations; the host repository is mounted read-only and the test
container has no network access. Docker may need to download images first.

With a configured local PyQGIS environment, `make test` runs against the source
tree. Both runners propagate failures and impose a timeout on hung tests.
The optional sample translation test is skipped when `i18n/af.qm` has not been
compiled; it is unrelated to the tracing compatibility checks.

## Resources and packaging

The checked-in `resources.py` works with either Qt binding. After changing
`resources.qrc` or its icon, use `make compile` (requires `pyrcc5` as a build
tool). The recipe rewrites the generated import to `qgis.PyQt`. Qt5 is not a
runtime dependency on QGIS 4. `pb_tool.cfg` ships this generated module without
recompiling it, and lists all tracing modules for deployment and packaging.

`make package` includes only the plugin runtime files listed in `pb_tool.cfg`
and compiled translations. It packages current edits, including uncommitted
ones, and does not deploy to a QGIS profile or upload anything.

## Desktop check before publishing

Use clean profiles in a Linux VM or a separate QGIS installation for both
generations. Install the ZIP through the plugin manager, load a representative
RGB raster and an editable MultiLineString layer, then:

1. Trace several segments, hover to preview, and finish with right-click.
2. Exercise `A`, `D`, `B`, `S`, `T` and Escape with canvas and layer-tree focus.
3. Check color and vector snapping, smoothing, and preview color/width.
4. Cancel a long trace, start another, and close/reopen the dock.
5. Save edits, reopen the project, and inspect the saved line geometry.

This checks desktop focus, rendering and responsiveness that the automated
suite cannot fully establish. No visible UI layout or translated text changed
as part of the compatibility migration.
