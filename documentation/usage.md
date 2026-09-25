# Using Raster Scribe

## Trace your first line

This example uses a small black line on a white raster and an empty vector
layer. Install Raster Scribe using the [README instructions](../README.md#install).

1. Copy the [example folder](example/) from this repository to a writable
   location, then open `example.qgs` in QGIS.
2. Select **Traced lines** in the Layers panel and click **Toggle Editing**
   (the pencil button).
3. Open **Raster → Raster Scribe → Raster Scribe**, or click the blue icon on
   the Raster Scribe toolbar.
4. Set **Layer to trace** to **Example raster**. Leave **Trace color** and both
   snapping options off. Leave **Preview path** on.
5. Click near the left end of the black line. Move the pointer along the line
   to its first bend. A pink preview shows the proposed path.
6. Click to add that segment. Continue along the line with a few more clicks.
7. Right-click to finish the line, then click **Save Layer Edits** in QGIS.

If a preview takes a wrong turn, move the pointer closer to the last point.
Use `B` to undo the last tracing step. You can repeat the example after
deleting the line from **Traced lines**.

While tracing, the accepted segments form a visible draft using QGIS's
digitizing appearance. Right-click adds the complete line to the vector layer
and its attribute table. QGIS Undo/Redo then removes or restores the whole line.

## Common tasks

### Trace your own map

Load an RGB raster and a MultiLineString or MultiCurve vector layer. Select the
vector layer in the Layers panel and click **Toggle Editing** (the pencil).
Activate Raster Scribe, then choose the raster under **Layer to trace**.
Trace and save edits as in the example above.

The raster, project, and vector layer may use different coordinate systems.
The plugin reads the raster's first three bands as red, green, and blue;
changes to QGIS display styling do not change those source values.

### Follow a particular color

Enable **Trace color** and choose the color to follow. You can also place the
pointer over a line and press `T`. This samples the pixel, enables **Trace
color**, and switches to tracing mode.

With **Trace color** off, each segment uses the color at its end point. This
can help when the line's color varies across the map.

### Join an existing line

Enable **Snap to vector layer**. Set a distance and click near a vertex in the
active vector layer or the current draft. The new segment ends at the nearest
vertex within that distance. Snapping joins coordinates; it does not merge
separate features.

The distance uses the map canvas's coordinate system: usually metres or feet
in a projected CRS, or degrees in a geographic CRS. It is not a screen-pixel
distance.

### Draw across a gap

Press `A` to draw straight segments. Click across the gap, then press `A`
again to resume tracing.

Press `D` for straight segments with extra vertices at 5-unit intervals in
the vector layer's coordinate system. A segment shorter than 5 units gets a
midpoint. Press `D` again to return to tracing.

### Cancel or finish a line

Press Escape to cancel the segment being calculated. Earlier segments remain.
Press `B` to cancel a pending segment, or remove the last accepted segment from
the current draft. Right-click to finish the line. After finishing, use QGIS's
Undo command (`Ctrl-Z`) to remove the entire line; Redo restores it. During
tracing, `B` affects the draft and QGIS Undo affects edits already in the layer.

Switching layers or tools, changing the project CRS, closing the dock, or saving
layer edits finishes the accepted draft. Discarding layer edits also discards
the draft. Finish with a right-click before saving if QGIS's save action is not
yet enabled. Changing trace settings or making other layer edits cancels a
pending calculation while keeping the draft.

With buffered transaction groups enabled, QGIS may finish the draft when
checking for unsaved edits.

## Controls and shortcuts

| Control | Purpose |
| --- | --- |
| Layer to trace | Choose the source raster. |
| Trace color | Follow the chosen color for every segment. |
| Snap to nearest | Move the end point toward the chosen color within the given radius, in raster pixels. Requires Trace color. Maximum radius: 99 pixels. |
| Snap to vector layer | Snap to a vertex in the active vector layer or current draft, within the given distance in canvas CRS units. Takes precedence over color snapping. |
| Smooth lines | Smooth the traced path. The preview uses the same smoothing. |
| Preview path | Show the proposed path while moving the pointer. The adjacent button sets its color. |
| Preview width | Set the width of the preview line. |

Color, snapping, smoothing, and preview preferences are saved between sessions.

| Key or mouse button | Action |
| --- | --- |
| Left-click | Start a line or add a segment. |
| Right-click | Finish the line. |
| `A` | Toggle straight-line mode. |
| `D` | Toggle straight-line mode with extra vertices. |
| `T` | Sample a color under the pointer and return to tracing. |
| `S` | Toggle color snapping. |
| `B` | Cancel pending work, undo the last tracing step, or remove the starting point. |
| Escape | Cancel the pending segment. |

Use shortcuts while the tracing tool is active. They also work with focus in
the Layers panel during a trace.

## Supported inputs

- A GDAL-readable raster with RGB values in bands 1–3. An RGB GeoTIFF is a
  useful starting point. Single-band grayscale and palette images need
  conversion to RGB before tracing.
- A north-up raster. Rotated, skewed, or flipped raster mappings are not supported.
- An editable MultiLineString or MultiCurve layer. Other geometry types,
  including ordinary LineString layers, are not accepted.

Pixels marked as nodata or invalid cannot be traced or sampled for color.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| Clicking does not start a line | Select the vector layer, enable editing, and check its geometry type. Choose a valid RGB raster in the dock. |
| The path follows the wrong line | Click closer together or sample the intended color with `T`. |
| A tracing limit is reached | Choose a shorter segment. Earlier segments remain intact. |
| No path is found | Check for gaps or invalid pixels. Use `A` to draw a straight segment across a gap. |
| Snapping reaches too far | Check the units: color snapping uses raster pixels; vector snapping uses the canvas CRS. A distance of 1 degree can be very large. |
| The raster is rejected | Check that GDAL can open it, that it has at least three bands, and that its mapping is north-up. |
| `B` does not undo an earlier trace | `B` only edits the current draft. Use QGIS's Undo command for finished lines. |

For a failure that persists, check the **Raster Scribe** tab in QGIS's Log
Messages panel. Include the message, QGIS version, and steps to reproduce it
when [reporting a problem](https://github.com/IongIer/raster_tracer/issues).
