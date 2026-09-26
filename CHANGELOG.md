# Changelog

## Unreleased

Changes from [Raster Tracer 0.3.3](https://github.com/mkondratyev85/raster_tracer):

### Tracing

- Show a path preview before clicking, with adjustable color and width.
- Add optional **Enhanced tracing (slower)** for contour continuity through
  interruptions, with a saved setting and configurable `E` shortcut. Switching
  cancels unaccepted work and refreshes the preview while keeping the draft.
- Save color, snapping, smoothing, preview, and finish-form preferences.
- Add `T` to sample the color under the pointer and `D` for dense straight lines.
- Make the drawn line match the preview, including smoothing and snapping.
- Optimize tracing and snapping, and reduce memory use on large rasters and
  long lines.
- Keep the unfinished line editable with `B`; undo and redo completed lines
  as a whole.
- Cancel pending traces when layers or tracing settings change, keeping earlier
  segments.
- Fix snapping to the nearest vector vertex.
- Skip nodata and invalid pixels when tracing and sampling colors.
- Ask for shorter segments when a trace exceeds the size limit.

### QGIS integration

- Support QGIS 3.40+ and QGIS 4.
- Allow installation alongside Raster Tracer, with separate preferences.
- Name the fork Raster Scribe and give it a blue icon.
- Place the plugin under the **Raster** menu.
- Add configurable controls with a preview of the selected bindings, saved
  shortcuts, and warnings about possible conflicts with current QGIS shortcuts.
- Allow clearing shortcuts, restoring defaults, and choosing or disabling the
  modifier that reverses the finish-form setting. Reject duplicate tracing keys.
- Optionally open the finished line's attribute form, with Shift+right-click by default
  reversing the saved choice once.
- Add fields to a layer from the feature form while preserving unfinished input.
- Create a temporary editable line layer using the selected raster's CRS.
- Add a Start tracing button in the panel, enabled when a suitable raster and
  editable line layer are selected.
