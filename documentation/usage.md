# Using Raster Scribe

For installation and tracing your own map, start with the [README](../README.md).
Keys below are defaults; **Controls…** shows your current bindings.

## Choose an output layer

Select your RGB raster under **Layer to trace**, then use **Create scratch
layer** for a temporary line layer. To use an existing layer, select it and
enable editing before clicking **Start tracing**. It must be **MultiLineString**
or **MultiCurve**; ordinary LineString layers are not accepted. A numeric
elevation field is an attribute, not 3D geometry.

## Convert a raster to RGB

Scribe reads RGB values from bands 1–3, not QGIS's display styling. Single-band
and indexed rasters must be converted first. Rotated, skewed and flipped raster
mappings are unsupported; the raster, project and vector layer may use different
coordinate systems. Scratch-layer creation requires a defined raster CRS.

For an indexed image with a palette, use **Processing Toolbox → GDAL → Raster
conversion → PCT to RGB** and save a GeoTIFF. This tool requires a color table.
See [PCT to RGB](https://docs.qgis.org/3.40/en/docs/user_manual/processing_algs/gdal/rasterconversion.html#pct-to-rgb).

For 8-bit grayscale values 0–255 without a palette, use a terminal with GDAL:

```sh
gdal_translate -of GTiff -b 1 -b 1 -b 1 -colorinterp red,green,blue "input.tif" "rgb.tif"
```

Other ranges, including black-and-white values 0–1, also need scaling; see
[`gdal_translate`](https://gdal.org/en/stable/programs/gdal_translate.html).
Load the converted file and select it under **Layer to trace**.

## Choose the color and tracing method

Enable **Trace color** to follow a chosen color, or press `T` over the intended
line to sample it. With the setting off, each segment uses its endpoint pixel's
color, so avoid placing endpoints on text or a neighboring contour.

**Enhanced tracing (slower)** (`E`) prefers continuity through interruptions.
It uses more processing and can still follow the wrong stroke or round a sharp
bend. Check the preview and use closer clicks or straight segments when needed.
**Smooth lines** smooths the output separately from the tracing method.

Switching methods cancels pending previews and clicks while preserving accepted
segments. Click again to accept the new preview. Enhanced tracing starts off by
default; your choice is saved. Straight-line modes stay selected when toggling it.

## Shortcuts and snapping

| Key or gesture | Action |
| --- | --- |
| Left-click / right-click | Add a segment / finish the line. |
| `A` | Toggle straight-line mode. |
| `D` | Toggle straight-line mode with extra vertices. |
| `E` | Toggle enhanced tracing. |
| `T` | Sample a color, enable Trace color, and return to tracing. |
| `N` | Toggle color snapping. |
| `B` | Cancel pending work or undo the last step in the current draft. |
| Escape | Cancel the pending segment. |
| Shift+right-click | Finish and reverse the attribute-form choice for this line. |

**Snap to nearest** uses raster pixels (up to 99) and requires **Trace color**.
**Snap to vector layer** uses canvas CRS units and takes precedence; it targets
vertices in the active layer or draft, without merging features. Dense mode
(`D`) adds vertices at most 5 vector-layer CRS units apart.

Preferences persist between sessions. In **Controls…**, click **Apply** to save
bindings or restored defaults. Duplicate keys are rejected; warnings flag
possible QGIS conflicts. Shortcuts work with tracing active and canvas focus,
or Layers-panel focus during a draft. They do not run while typing in forms.

## Attributes and unfinished lines

Enable **Open attributes after finishing** to enter values after each line.
Shift+right-click reverses the choice once; change or disable that modifier in
**Controls…**. Cancelling the form keeps the line but discards unaccepted values.

Use **Add field to layer…** in the form for missing fields. New fields belong
to the whole layer and remain if you cancel the form. Custom form layouts may
need the field added through QGIS's form settings. Required fields may need
default values, because the form opens after line creation.

Before finishing, `B` edits the draft; afterward, QGIS Undo/Redo affects the
whole line. Changing layers, tools or project CRS, closing the dock, or saving
edits finishes the accepted draft without opening a form. Discarding edits
discards it. Scratch data remains temporary until **Make Permanent** is used.

## Common problems

| Problem | What to try |
| --- | --- |
| Start tracing is disabled | Check its tooltip for the missing raster, layer or editing requirement. |
| Wrong route | Use closer clicks, sample with `T`, or try `E` through a crossing. |
| No path or a tracing limit | Use a shorter segment. Nodata pixels cannot be traced; use `A` across gaps. |
| A shortcut does not respond | Check Controls… for its binding and conflicts, then focus the canvas. |
| Snapping reaches too far | Check its units, especially when the canvas CRS uses degrees. |

For a persistent problem, include **Log Messages → Raster Scribe** output and
your QGIS version when [reporting it](https://github.com/IongIer/raster_scribe/issues).
