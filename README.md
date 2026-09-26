# Raster Scribe

Raster Scribe is a QGIS plugin for tracing lines in RGB raster maps. Click
along a contour, road, or other line, and the plugin follows pixels of a
similar color to draw vector geometry.

This is an independently maintained fork of
[Raster Tracer](https://github.com/mkondratyev85/raster_tracer), originally
developed by Mikhail Kondratyev. The fork is maintained by
[IongIer](https://github.com/IongIer).

## Differences from Raster Tracer

- Live path previews, with adjustable color and width.
- Optional enhanced tracing for interrupted contours, toggled with `E`.
- Trace large rasters without loading the whole image into memory.
- Saved color, snapping, smoothing, preview, and finish-form preferences.
- Shortcuts for sampling a color and drawing a straight line with extra vertices.
- Configurable controls with QGIS shortcut conflict warnings, and an optional
  attribute form after finishing each line.
- Add fields from the attribute form, and create a scratch line layer for a raster.
- Cancel pending traces or undo the last segment while keeping earlier work.
- Support for QGIS 3.40+ and QGIS 4, using the same package.

See [the changelog](CHANGELOG.md) for details.

![Raster Scribe tracing the example in QGIS 4](documentation/raster-scribe.png)

The blue line is drawn geometry; the pink line is the next segment's preview.

## Install

Build the plugin from the repository root with Python 3:

```sh
python3 scripts/package-plugin.py
```

In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP** and
select the ZIP created in `dist/`. Open Raster Scribe from the **Raster** menu
or its blue toolbar icon.

Raster Scribe can be installed alongside Raster Tracer. Each plugin keeps
its own preferences. Saved vector layers need no conversion.

## Use

Load an RGB raster, activate Raster Scribe, and choose it under **Layer to
trace**. Click **Create scratch layer** to start with a temporary line layer,
or select an existing MultiLineString or MultiCurve vector layer and click
**Toggle Editing** (the pencil), then **Start tracing** in the plugin panel.
Click a starting point, move along the line to see the preview, and click to
add a segment. Right-click to finish, then click **Save Layer Edits** in QGIS.

Enable **Open attributes after finishing** to enter values after each line;
Shift+right-click reverses that choice for one line by default. Use **Add field to layer…**
in the form if a field is missing. Scratch layers are temporary: use QGIS's
**Make Permanent** action to keep their data after closing the project.

Open **Controls…** to try shortcut customization, review possible QGIS conflicts,
or change the modifier used with right-click. Choose **Apply** to save changes
for your QGIS user profile; **Restore Defaults**, then **Apply**, restores the
original controls.

[Trace your first line](documentation/usage.md#trace-your-first-line) using the
small example included in this repository. The guide also covers
[tasks](documentation/usage.md#common-tasks),
[controls](documentation/usage.md#controls-and-shortcuts), and
[troubleshooting](documentation/usage.md#troubleshooting).

Tested on Linux and Windows. macOS has not been tested.

## Development and support

- [Build and development](documentation/development.md)
- [Testing guide](test/README.md)
- [Report a problem](https://github.com/IongIer/raster_tracer/issues)

Maintenance follows the maintainer's own use of the plugin. Report bugs on
this fork's issue tracker, with the QGIS version and steps to reproduce them.

## License and credits

Raster Scribe as a combined plugin is licensed under **GNU GPL version 3 or
later (GPL-3.0-or-later)**. See [LICENSE](LICENSE) for the full text and
[NOTICE.txt](NOTICE.txt) for component attribution and licensing details.

Raster Tracer's original MIT notice is preserved in NOTICE.txt, alongside
HamiltonFastMarching's original notices and attribution. Inherited GPL version
2 or later source headers are retained; the upstream grants remain attached
to their respective material.

Enhanced tracing adapts Jean-Marie Mirebeau's HamiltonFastMarching solver in
Python under GPL version 3 or later and uses QGIS's NumPy.

The blue icon is adapted from Raster Tracer's original icon.
