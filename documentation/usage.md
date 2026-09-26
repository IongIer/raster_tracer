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
   snapping options off. Leave **Preview path** on, then click **Start tracing**.
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
Click **Start tracing** in the plugin panel to activate the tracing tool.
Alternatively, choose the raster and click **Create scratch layer** to create
a temporary vector layer ready for tracing. Trace and save edits as in the
example above.

**Start tracing** is enabled when a suitable raster is selected and the active
vector layer is an editable MultiLineString or MultiCurve layer. Use it again
after switching to another QGIS tool, such as pan or identify.

The raster, project, and vector layer may use different coordinate systems.
The plugin reads the raster's first three bands as red, green, and blue;
changes to QGIS display styling do not change those source values.

### Create a temporary line layer

Choose the source raster under **Layer to trace**, then click **Create scratch
layer**. The plugin adds a 2D MultiLineString layer using the raster's CRS,
selects it, and enables editing. The button is disabled without a suitable
raster. A raster with an undefined CRS needs a CRS assigned before creating
the layer.

The new layer has no attribute fields. You can add the fields you need from
the attribute form described below, or through QGIS's layer tools. A numeric
field such as `z` is an ordinary attribute; it does not make the geometry 3D.

Scratch layers are temporary. **Save Layer Edits** and saving the project do
not make their data permanent. Use QGIS's **Make Permanent** action for the
layer to save it to a file before closing the project.

### Enter attributes after each line

Enable **Open attributes after finishing** to open the new line's attribute
form after right-clicking to finish. Enter values and accept the form to
apply them. The checkbox is off by default and is saved between sessions.
Shift+right-click reverses the saved choice for that finish only:

| Open attributes after finishing | Right-click | Shift+right-click |
| --- | --- | --- |
| Off | Finish the line. | Finish and open its form. |
| On | Finish and open its form. | Finish the line. |

Cancelling the form keeps the finished line and discards unaccepted attribute
values. Line creation and accepted attribute edits are separate undoable
edits. The form does not save all layer edits; use QGIS's normal save action
when ready.

Only an explicit finish gesture can open this form. Finishing a draft while
saving, changing layers or tools, or closing the plugin does not open it.
An empty draft or an unsuccessful finish does not open a form either.

#### Add a missing field

Click **Add field to layer…** in the form to define a field name and supported
type, with length or precision where applicable. For example, add a decimal
field named `z` for an elevation entered manually, or a text field for notes.
The plugin does not require any particular field name.

Adding a field changes the whole layer. Existing features normally have no
value in it until you populate them. The form refreshes after the addition
while keeping values you have already typed. Confirmed field creation is a
separate undoable layer edit and remains even if you cancel the feature form.

Field names and types must satisfy the layer's data provider. The button is
disabled if the layer cannot add fields. A custom form layout may require
placing the new field in the layout through QGIS's form configuration; the
plugin does not rearrange a custom layout.

Some providers enforce required values as soon as a feature is added. This
form opens after line creation, so such layers may need suitable defaults
before tracing.

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

Press `D` for straight segments with evenly spaced extra vertices, at most
5 units apart in the vector layer's coordinate system. A segment shorter than
5 units gets a midpoint. Press `D` again to return to tracing.

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
| Create scratch layer | Create and select a temporary editable line layer with the raster's CRS. |
| Start tracing | Activate the tracing tool for the selected raster and active editable line layer. |
| Trace color | Follow the chosen color for every segment. |
| Snap to nearest | Move the end point toward the chosen color within the given radius, in raster pixels. Requires Trace color. Maximum radius: 99 pixels. |
| Snap to vector layer | Snap to a vertex in the active vector layer or current draft, within the given distance in canvas CRS units. Takes precedence over color snapping. |
| Smooth lines | Smooth the traced path. The preview uses the same smoothing. |
| Preview path | Show the proposed path while moving the pointer. The adjacent button sets its color. |
| Preview width | Set the width of the preview line. |
| Open attributes after finishing | Open the new feature's form after finishing with right-click. Shift+right-click reverses the choice once. |
| Controls… | Show the plugin's keyboard and mouse controls. |

Color, snapping, smoothing, preview, and **Open attributes after finishing**
preferences are saved between sessions.

| Key or mouse button | Action |
| --- | --- |
| Left-click | Start a line or add a segment. |
| Right-click | Finish the line; open its form if Open attributes after finishing is enabled. |
| Shift+right-click | Finish the line and reverse the form-opening choice for this line only. |
| `A` | Toggle straight-line mode. |
| `D` | Toggle straight-line mode with extra vertices. |
| `T` | Sample a color under the pointer and return to tracing. |
| `S` | Toggle color snapping. |
| `B` | Cancel pending work, undo the last tracing step, or remove the starting point. |
| Escape | Cancel the pending segment. |

Use shortcuts while the tracing tool is active. They also work with focus in
the Layers panel during a trace. Letter shortcuts use the unmodified key and
do not run while typing in the attribute form or other input fields. Bindings
are fixed in this version. A matching shortcut assigned in QGIS may take
precedence; check **Settings → Keyboard Shortcuts** if a plugin key does not
respond.

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
| Start tracing is disabled | Select a MultiLineString or MultiCurve vector layer, enable editing, and choose a valid RGB raster in the dock. |
| Clicking does not start a line | Check the source raster and editable vector layer, then click Start tracing in the dock. |
| The path follows the wrong line | Click closer together or sample the intended color with `T`. |
| A tracing limit is reached | Choose a shorter segment. Earlier segments remain intact. |
| No path is found | Check for gaps or invalid pixels. Use `A` to draw a straight segment across a gap. |
| Snapping reaches too far | Check the units: color snapping uses raster pixels; vector snapping uses the canvas CRS. A distance of 1 degree can be very large. |
| The raster is rejected | Check that GDAL can open it, that it has at least three bands, and that its mapping is north-up. |
| `B` does not undo an earlier trace | `B` only edits the current draft. Use QGIS's Undo command for finished lines. |
| The attribute form does not open | Enable Open attributes after finishing, or use Shift+right-click with it disabled. A line must be created successfully. Automatic finishes during saving or tool changes do not open forms. |
| A new field is missing from the form | Check whether the layer uses a custom form layout and add the field to that layout in the layer's form configuration. |
| A scratch layer is missing after reopening the project | Scratch layers are temporary. Use Make Permanent before closing the project to keep their data. |

For a failure that persists, check the **Raster Scribe** tab in QGIS's Log
Messages panel. Include the message, QGIS version, and steps to reproduce it
when [reporting a problem](https://github.com/IongIer/raster_tracer/issues).
