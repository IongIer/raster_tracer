# Testing

Run commands from the repository root. See [Development](../documentation/development.md)
for lint and formatting checks.

## Run locally

Core search and smoothing tests need ordinary Python, without QGIS or NumPy:

```sh
python3 scripts/run-tests.py --core-only
```

For the full source-tree suite, use a Python interpreter configured for PyQGIS:

```sh
make test PYTHON=/path/to/qgis/python
```

Do not use `uv run make test`; the tooling environment has no QGIS runtime.

## Test the packaged plugin

Build the ZIP, then run it in QGIS 4:

```sh
python3 scripts/package-plugin.py --output dist/test-plugin.zip

docker run --rm --network none \
  --tmpfs /tmp:rw,exec,nosuid,size=512m \
  -e QT_QPA_PLATFORM=offscreen -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --workdir /workspace --entrypoint python3 \
  qgis/qgis:4.0-trixie \
  scripts/run-tests.py --plugin-zip dist/test-plugin.zip \
  --expected-qgis 4.0 --expected-qt-major 6
```

Repeat with each image and matching version arguments:

| Image | `--expected-qgis` | `--expected-qt-major` |
| --- | --- | --- |
| `qgis/qgis:3.40-bookworm` | `3.40` | `5` |
| `qgis/qgis:3.44-bookworm` | `3.44` | `5` |
| `qgis/qgis:4.0-trixie` | `4.0` | `6` |

The suite loads the ZIP in real QGIS. The temporary filesystem avoids slow
QSettings cleanup on Docker's writable layer. CI's pinned image digests are
in the [workflow](../.github/workflows/compatibility.yml).

The runner prints runtime versions, fails on errors or version mismatches,
and stops after 180 seconds if it hangs. Report skipped tests separately.
The translation test skips without `test/af.qm`; rebuild that fixture with
`lrelease test/af.ts -qm test/af.qm`.

Automated tests run headlessly; check the affected workflow in QGIS for
appearance, responsiveness and input behavior.
