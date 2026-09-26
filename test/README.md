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
Check the desktop workflow and appearance as well as the automated results.
Use default bindings for the tracing checks unless a row calls for customization.

| Action | Expected result |
| --- | --- |
| Follow [Trace your first line](../documentation/usage.md#trace-your-first-line), including enabling editing | The line follows the raster, matches the preview, and stays after finishing. |
| Select an existing line layer and source raster, enable editing, and click **Start tracing** below **Create scratch layer** | The canvas enters tracing mode and accepts a starting point without another toolbar click or a color sample. |
| Clear the source raster, select an unsupported layer, and stop or start editing | **Start tracing** is enabled only with a suitable raster and an active editable MultiLineString or MultiCurve layer; its tooltip explains missing prerequisites. |
| Switch to pan or identify, then click **Start tracing** again; also click it during an active draft | Tracing resumes with the same settings; clicking while tracing keeps the draft intact. |
| Use `A`, `D`, `T`, and `N`; repeat with Layers-panel focus during a trace | Modes, sampled color, and color snapping change as described in the usage guide. |
| Toggle **Enhanced tracing (slower)** with its checkbox and `E`, including while a preview or clicked segment is calculating | The checkbox and saved choice agree; old work cannot draw or commit; accepted segments stay in place and a new preview uses the selected method. Straight/dense modes stay selected. |
| Open **Controls…** and compare the listed controls with the tool | Every keyboard and mouse action is described, including the chosen finish-form modifier and the focus requirement. |
| Assign a modified key in **Controls…**, inspect the preview, and click **Apply** | The preview describes the proposed key; after Apply, only the new combination runs that action on the canvas and in the Layers panel during a draft. |
| Clear a shortcut and apply; then restore defaults and apply | Clearing disables that action's keyboard shortcut; restoring defaults brings it back. |
| Assign the same combination to two tracing actions | Both entries show a red Error message and Apply is disabled until it is resolved. |
| Edit a binding and close without applying; reopen Controls | The previous saved binding remains active and is shown again. |
| Apply changed bindings and a finish modifier, reopen the dock, switch projects, and restart QGIS | The same QGIS user profile retains the chosen settings. |
| In QGIS Keyboard Shortcuts, assign a tracing key or a longer sequence beginning with it; change that assignment while Controls remains open | The amber Warning message identifies the current owner and sequence, including first-key conflicts; it updates when the QGIS assignment changes. Apply remains available for valid tracing bindings. |
| Apply a binding with a QGIS conflict warning | QGIS's existing assignment stays unchanged; check which action receives the key in the actual tracing context. Disabled or widget-specific actions may not interfere. |
| Try extra modifiers on configured keys on the canvas and in the Layers panel | Only the exact configured keyboard combination triggers the tracing action. |
| Press Escape during a long trace; start another | Earlier segments remain and the new trace can finish. |
| Press `B` during pending work, then after an accepted segment | Pending work cancels; the last draft segment can be removed. |
| Finish a multi-segment line, then use QGIS Undo/Redo | The whole finished line disappears and returns as one edit. |
| Pan and zoom while tracing, then snap back to the draft | The draft stays visible and its vertices remain available for snapping. |
| Enable both snapping options and smoothing | The endpoint joins the intended vertex; preview and saved geometry agree. |
| Change the raster, vector layer, or CRS during a trace | Earlier geometry remains and a late result does not add another segment. |
| Close and reopen the dock during a trace; unload while work is pending | The dock and tool close without a crash or stray result. |
| Change preview color and width; reopen the dock | The saved appearance is restored. |
| Toggle **Open attributes after finishing**, reopen the dock, and restart QGIS | The chosen setting is restored. |
| Finish a line with right-click and Shift+right-click, with the form option both on and off | The form opens according to the checkbox; Shift reverses it for one finish without changing the saved setting. |
| Enter attribute values and accept the form; undo and redo | Values are applied to the new line. Attribute edits and line creation can be undone independently. |
| Type letters used by default and customized tracing shortcuts into attribute forms and other input fields | Text is entered normally and tracing modes do not change. |
| Change the finish modifier, test it with the form checkbox on and off, then disable the override | Holding the chosen modifier reverses the form choice once, including when another modifier is held; Disabled leaves the checkbox in control. The saved checkbox does not change. |
| Test alternative finish modifiers using the desktop's mouse or trackpad | The intended gesture reaches the tracing tool; check for operating-system interception, which the keyboard conflict warnings cannot detect. |
| Type values, add a field, and continue editing | Typed values survive the refresh; the new field is available in an autogenerated form. |
| Add decimal and text fields; try duplicate names and a layer that cannot add fields | Valid fields are added to the whole layer; unsupported additions fail clearly or are disabled. |
| Cancel a form with unaccepted values, including after adding a field | The finished line and confirmed field remain; the unaccepted values do not. Field creation is a separate undoable edit. |
| Use a layer with a custom form layout and add a field | The custom layout is preserved; a field excluded by the layout can be added through form configuration. |
| Save edits, change layers or tools, or close the dock with the form option enabled | Drafts finish as before without opening a form. |
| Right-click an empty draft or finish while a segment is pending | No empty-feature form opens; accepted geometry is finished without a late calculation adding another segment. |
| Select a raster and click **Create scratch layer** | A 2D MultiLineString layer matches the raster CRS, is selected, is editable, and has no attribute fields. |
| Clear the source raster or use a raster with an undefined CRS | Scratch-layer creation is disabled without a suitable raster; an undefined CRS is handled explicitly. |
| Create a scratch layer, trace, add a numeric field, enter values for several lines, and occasionally bypass the form | The complete tracing and attribute-entry workflow is usable; focus returns to tracing after forms close. |
| Use **Make Permanent** for the scratch layer, save, and reopen the project | Geometry, fields, and values remain in the permanent layer. |
| Save edits and reopen the project | The saved line remains in the vector layer. |
| Trace a representative large raster | Hover feedback, cancellation, and panning remain usable. |
| On QGIS 3, load Raster Tracer too and switch between the tools | Their names, icons, settings, and shortcuts remain separate; unloading either leaves the other usable. |

Record the tested commit or working-tree changes, QGIS version, platform,
and any failures. Keep dated test reports separate from this guide.

macOS input behavior remains unverified until tested on a desktop. In a macOS
run, include secondary-click and the default and customized finish modifiers
using both a mouse and the available trackpad gesture. Verify native modifier
labels and customized keyboard combinations too.
