# Changelog

## Unreleased — Raster Scribe fork

Changes since upstream Raster Tracer 0.3.3:

### Tracing

- Show a path preview before clicking, with adjustable color and width.
- Read raster windows instead of loading the whole image for tracing.
- Save color, snapping, smoothing, and preview preferences.
- Add `T` to sample the color under the pointer and `D` for dense straight lines.
- Make the drawn line match the preview, including smoothing and snapping.
- Cancel pending traces when layers or tracing settings change, keeping earlier
  segments.
- Prevent `B` from undoing unrelated edits.
- Fix snapping to the nearest vector vertex.
- Skip nodata and invalid pixels when tracing and sampling colors.
- Ask for shorter segments when a trace exceeds the size limit.

### Compatibility

- Support QGIS 3.40+ and QGIS 4.
- Allow installation alongside Raster Tracer, with separate preferences.

### Name and appearance

- Name the fork Raster Scribe and give it a blue icon.
- Place the plugin under the **Raster** menu.

## Upstream history

These releases belong to [Raster Tracer](https://github.com/mkondratyev85/raster_tracer).

- **0.3.3:** Fix QGIS 3.30 compatibility.
- **0.3.2:** Fix tracing with different raster and vector coordinate systems.
- **0.3.1:** Add vector snapping and fix dock closing.
- **0.3.0:** Run tracing in the background.
- **0.2.0:** Make smoothing optional and warn about unsupported vector layers.
- **0.1.1:** Update plugin details and homepage.
- **0.1:** Initial release.
