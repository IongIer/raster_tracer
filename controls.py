"""Fixed tracing bindings and their user-facing reference."""

from html import escape

from qgis.PyQt.QtCore import QCoreApplication, Qt
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)


class RasterScribeControls:
    """Keep translation extraction and runtime lookup in the same context."""

    @staticmethod
    def tr(message):
        return QCoreApplication.translate("RasterScribeControls", message)


SHORTCUT_ACTIONS = {
    Qt.Key.Key_A: "toggle_straight",
    Qt.Key.Key_D: "toggle_dense_straight",
    Qt.Key.Key_T: "sample_color",
    Qt.Key.Key_S: "toggle_color_snap",
    Qt.Key.Key_B: "undo_segment",
    Qt.Key.Key_Escape: "cancel_pending",
}


def shortcut_action(event):
    """Match only a bare key, consistently on the canvas and in the Layers panel."""
    if event.modifiers() != Qt.KeyboardModifier.NoModifier:
        return None
    return SHORTCUT_ACTIONS.get(event.key())


def control_rows():
    """Translate when displayed, after the plugin's translator is installed."""
    descriptions = {
        "toggle_straight": RasterScribeControls.tr("Toggle straight-line mode."),
        "toggle_dense_straight": RasterScribeControls.tr(
            "Toggle straight-line mode with extra vertices."
        ),
        "sample_color": RasterScribeControls.tr(
            "Sample a color under the pointer and return to tracing.",
        ),
        "toggle_color_snap": RasterScribeControls.tr("Toggle color snapping."),
        "undo_segment": RasterScribeControls.tr(
            "Cancel pending work, undo the last tracing step, or remove the starting point.",
        ),
        "cancel_pending": RasterScribeControls.tr("Cancel the pending segment."),
    }
    mouse_rows = (
        (
            RasterScribeControls.tr("Left-click"),
            RasterScribeControls.tr("Start a line or add a segment."),
        ),
        (
            RasterScribeControls.tr("Right-click"),
            RasterScribeControls.tr(
                "Finish the line. Open its form if Open attributes after finishing is enabled.",
            ),
        ),
        (
            RasterScribeControls.tr("Shift+right-click"),
            RasterScribeControls.tr(
                "Finish the line and reverse the form setting for this line only.",
            ),
        ),
    )
    key_rows = tuple(
        (
            QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText),
            descriptions[action],
        )
        for key, action in SHORTCUT_ACTIONS.items()
    )
    return mouse_rows + key_rows


def create_controls_dialog(parent):
    """Create a nonmodal reference without registering any shortcuts."""
    dialog = QDialog(parent)
    dialog.setObjectName("rasterScribeControlsDialog")
    dialog.setWindowTitle(RasterScribeControls.tr("Raster Scribe controls"))
    layout = QVBoxLayout(dialog)
    browser = QTextBrowser(dialog)
    browser.setObjectName("controlsReference")
    scope = RasterScribeControls.tr(
        "Use these keys with the tracing tool active and focus on the map canvas. "
        "During a trace they also work in the Layers panel. "
        "Typing in forms and other input fields does not trigger tracing actions.",
    )
    modifiers = RasterScribeControls.tr(
        "Use keys without modifiers. A QGIS shortcut assigned to the same key may take priority.",
    )
    rows = "".join(
        "<tr><td><b>{}</b></td><td>{}</td></tr>".format(
            escape(control), escape(description)
        )
        for control, description in control_rows()
    )
    browser.setHtml(
        "<p>{}</p><table cellspacing='0' cellpadding='6'>{}</table><p>{}</p>".format(
            escape(scope), rows, escape(modifiers)
        )
    )
    layout.addWidget(browser)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(650, 580)
    return dialog
