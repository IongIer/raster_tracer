"""Editable tracing shortcuts and their current user-facing reference."""

from html import escape

from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt, QTimer
from qgis.PyQt.QtGui import QKeySequence, QPalette
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QScrollArea,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .shortcut_conflicts import find_shortcut_conflicts
from .shortcuts import (
    DEFAULT_BINDINGS,
    FINISH_MODIFIERS,
    ShortcutSettings,
    action_descriptions,
    modifier_label,
    sequence_label,
    validation_errors,
)


class RasterScribeControls:
    """Keep translation extraction and runtime lookup in the same context."""

    @staticmethod
    def tr(message):
        return QCoreApplication.translate("RasterScribeControls", message)


def control_rows(bindings=None, finish_modifier="Shift"):
    """Describe the proposed or saved keys, including the mouse modifier."""
    bindings = DEFAULT_BINDINGS if bindings is None else bindings
    rows = [
        (
            RasterScribeControls.tr("Left-click"),
            RasterScribeControls.tr("Start a line or add a segment."),
        ),
        (
            RasterScribeControls.tr("Right-click"),
            RasterScribeControls.tr(
                "Finish the line. Open its form if Open attributes after finishing is enabled."
            ),
        ),
    ]
    if finish_modifier != "None":
        rows.append(
            (
                RasterScribeControls.tr("{modifier}+right-click").format(
                    modifier=modifier_label(finish_modifier)
                ),
                RasterScribeControls.tr(
                    "Finish the line and reverse the form setting for this line only."
                ),
            )
        )
    descriptions = action_descriptions()
    rows.extend(
        (
            sequence_label(bindings[action])
            if bindings[action]
            else RasterScribeControls.tr("Unassigned"),
            descriptions[action],
        )
        for action in DEFAULT_BINDINGS
    )
    return tuple(rows)


class SingleStrokeKeySequenceEdit(QKeySequenceEdit):
    """Capture one combination consistently with both Qt5 and Qt6."""

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        if event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().event(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key not in (
            Qt.Key.Key_unknown,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_AltGr,
        ):
            modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
            mask = modifiers.value if hasattr(modifiers, "value") else int(modifiers)
            self.setKeySequence(QKeySequence(mask | key))
        event.accept()

    def keyReleaseEvent(self, event):
        # The base class collects up to four strokes on Qt5. We already recorded
        # the complete combination on key press and do not start its timer.
        event.accept()


class ControlsDialog(QDialog):
    """Preview and save plugin-local bindings without changing QGIS shortcuts."""

    def __init__(self, parent, shortcuts=None, main_window=None):
        super().__init__(parent)
        self.setObjectName("rasterScribeControlsDialog")
        self.setWindowTitle(RasterScribeControls.tr("Raster Scribe controls"))
        self.shortcuts = shortcuts if shortcuts is not None else ShortcutSettings(self)
        self.main_window = (
            main_window
            if main_window is not None
            else parent.window()
            if parent is not None
            else None
        )
        self._loading = False
        self._reference_rows = None
        self.editors = {}
        self.conflict_labels = {}
        layout = QVBoxLayout(self)
        scope = QLabel(
            RasterScribeControls.tr(
                "Use these keys with the tracing tool active and focus on the map canvas. "
                "During a trace they also work in the Layers panel. "
                "Typing in forms and other input fields does not trigger tracing actions."
            ),
            self,
        )
        scope.setWordWrap(True)
        layout.addWidget(scope)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        body_layout = QVBoxLayout(body)
        keyboard_group = QGroupBox(RasterScribeControls.tr("Keyboard shortcuts"), body)
        grid = QGridLayout(keyboard_group)
        grid.setColumnStretch(0, 1)
        descriptions = action_descriptions()
        for row, action in enumerate(DEFAULT_BINDINGS):
            label = QLabel(descriptions[action], keyboard_group)
            label.setWordWrap(True)
            editor = SingleStrokeKeySequenceEdit(keyboard_group)
            editor.setObjectName("shortcut_" + action)
            editor.setAccessibleName(descriptions[action])
            editor.setToolTip(
                RasterScribeControls.tr(
                    "Press one key combination. Use Clear to unassign it."
                )
            )
            editor.setMinimumWidth(155)
            label.setBuddy(editor)
            clear = QToolButton(keyboard_group)
            clear.setText(RasterScribeControls.tr("Clear"))
            clear.setAccessibleName(
                RasterScribeControls.tr("Clear shortcut: {action}").format(
                    action=descriptions[action]
                )
            )
            clear.clicked.connect(editor.clear)
            warning = QLabel(keyboard_group)
            warning.setObjectName("shortcutWarning_" + action)
            warning.setWordWrap(True)
            warning.setTextFormat(Qt.TextFormat.RichText)
            warning.hide()
            grid.addWidget(label, row * 2, 0)
            grid.addWidget(editor, row * 2, 1)
            grid.addWidget(clear, row * 2, 2)
            grid.addWidget(warning, row * 2 + 1, 0, 1, 3)
            self.editors[action] = editor
            self.conflict_labels[action] = warning
            editor.keySequenceChanged.connect(self.refresh)
        body_layout.addWidget(keyboard_group)
        finish_group = QGroupBox(RasterScribeControls.tr("Finishing a line"), body)
        finish_layout = QVBoxLayout(finish_group)
        modifier_row = QHBoxLayout()
        modifier_label_widget = QLabel(
            RasterScribeControls.tr("Reverse the form setting for one line:"),
            finish_group,
        )
        modifier_label_widget.setWordWrap(True)
        modifier_row.addWidget(modifier_label_widget, 1)
        self.finish_modifier_combo = QComboBox(finish_group)
        self.finish_modifier_combo.setObjectName("finishModifier")
        for modifier in FINISH_MODIFIERS:
            label = (
                RasterScribeControls.tr("Disabled")
                if modifier == "None"
                else RasterScribeControls.tr("{modifier}+right-click").format(
                    modifier=modifier_label(modifier)
                )
            )
            self.finish_modifier_combo.addItem(label, modifier)
        modifier_label_widget.setBuddy(self.finish_modifier_combo)
        modifier_row.addWidget(self.finish_modifier_combo)
        finish_layout.addLayout(modifier_row)
        mouse_note = QLabel(
            RasterScribeControls.tr(
                "This mouse modifier is separate from keyboard shortcuts. "
                "QGIS keyboard shortcuts do not reserve modifier keys by themselves. "
                "Your operating system or window manager may still intercept some mouse combinations."
            ),
            finish_group,
        )
        mouse_note.setWordWrap(True)
        finish_layout.addWidget(mouse_note)
        self.finish_modifier_combo.currentIndexChanged.connect(self.refresh)
        body_layout.addWidget(finish_group)
        reference_label = QLabel(RasterScribeControls.tr("Controls preview"), body)
        body_layout.addWidget(reference_label)
        self.browser = QTextBrowser(body)
        self.browser.setObjectName("controlsReference")
        self.browser.setMinimumHeight(170)
        body_layout.addWidget(self.browser)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.status_label = QLabel(self)
        self.status_label.setObjectName("shortcutStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(self.apply_changes)
        self.buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(self.restore_defaults)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.conflict_timer = QTimer(self)
        self.conflict_timer.setInterval(1500)
        self.conflict_timer.timeout.connect(self.refresh)
        self.shortcuts.changed.connect(self.load_saved)
        self.load_saved()
        self.resize(790, 740)

    def proposed_bindings(self):
        return {
            action: editor.keySequence().toString(
                QKeySequence.SequenceFormat.PortableText
            )
            for action, editor in self.editors.items()
        }

    def set_proposed(self, bindings, finish_modifier):
        self._loading = True
        try:
            for action, editor in self.editors.items():
                editor.setKeySequence(QKeySequence.fromString(bindings[action]))
            self.finish_modifier_combo.setCurrentIndex(
                self.finish_modifier_combo.findData(finish_modifier)
            )
        finally:
            self._loading = False
        self.refresh()

    def load_saved(self):
        self.set_proposed(self.shortcuts.bindings, self.shortcuts.finish_modifier)

    def restore_defaults(self):
        self.set_proposed(DEFAULT_BINDINGS, "Shift")

    def set_feedback(self, label, messages):
        """Keep severity readable in light/dark themes and without color vision."""
        background = label.parentWidget().palette().color(QPalette.ColorRole.Window)
        dark = background.lightness() < 128
        colors = {
            "error": "#ffb4ab" if dark else "#b42318",
            "warning": "#ffd580" if dark else "#8a4b00",
        }
        html, plain = [], []
        for severity, message in messages:
            if severity == "error":
                message = RasterScribeControls.tr("Error: {message}").format(
                    message=message
                )
            elif severity == "warning":
                message = RasterScribeControls.tr("Warning: {message}").format(
                    message=message
                )
            plain.append(message)
            text = escape(message)
            if severity in colors:
                text = '<span style="color: {}; font-weight: 600">{}</span>'.format(
                    colors[severity], text
                )
            html.append(text)
        label.setText("<br>".join(html))
        label.setAccessibleName("\n".join(plain))

    def refresh(self, *_args):
        if self._loading:
            return
        bindings = self.proposed_bindings()
        modifier = self.finish_modifier_combo.currentData()
        errors = validation_errors(bindings, modifier)
        conflicts = find_shortcut_conflicts(bindings, self.main_window)
        for action, label in self.conflict_labels.items():
            messages = []
            if action in errors:
                messages.append(("error", errors[action]))
            for conflict in conflicts.get(action, ()):
                message = RasterScribeControls.tr(
                    'Possible QGIS conflict: "{owner}" ({shortcut}).'
                ).format(owner=conflict.owner, shortcut=conflict.shortcut)
                if conflict.partial:
                    message += " " + RasterScribeControls.tr(
                        "This is the first key of a multi-key QGIS shortcut."
                    )
                if not conflict.enabled:
                    message += " " + RasterScribeControls.tr(
                        "Currently disabled in QGIS."
                    )
                messages.append(("warning", message))
            self.set_feedback(label, messages)
            label.setVisible(bool(messages))
        rows = "".join(
            "<tr><td><b>{}</b></td><td>{}</td></tr>".format(
                escape(control), escape(description)
            )
            for control, description in control_rows(bindings, modifier)
        )
        if rows != self._reference_rows:
            self.browser.setHtml(
                "<table cellspacing='0' cellpadding='3'>{}</table>".format(rows)
            )
            self._reference_rows = rows
        dirty = (
            bindings != self.shortcuts.bindings
            or modifier != self.shortcuts.finish_modifier
        )
        self.apply_button.setEnabled(dirty and not errors)
        if errors:
            severity, status = (
                "error",
                RasterScribeControls.tr(
                    "Resolve invalid or duplicate tracing shortcuts before applying."
                ),
            )
        elif any(conflicts.values()):
            severity, status = (
                "warning",
                RasterScribeControls.tr(
                    "QGIS may take priority for the listed shortcuts. Choose other combinations "
                    "or apply them anyway. Changes take effect when you choose Apply."
                ),
            )
        elif dirty:
            severity, status = (
                "",
                RasterScribeControls.tr("Changes take effect when you choose Apply."),
            )
        else:
            severity, status = (
                "",
                RasterScribeControls.tr("These controls are currently saved."),
            )
        self.set_feedback(self.status_label, [(severity, status)])

    def apply_changes(self):
        try:
            self.shortcuts.apply(
                self.proposed_bindings(), self.finish_modifier_combo.currentData()
            )
        except ValueError as error:
            self.set_feedback(self.status_label, [("error", str(error))])
            return
        self.refresh()

    def showEvent(self, event):
        self.load_saved()
        self.conflict_timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self.conflict_timer.stop()
        super().hideEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if not hasattr(self, "conflict_timer"):
            return
        if event.type() in (
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        ):
            # Child palettes update after the dialog receives the theme change.
            QTimer.singleShot(0, self.refresh)
        elif event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.refresh()


def create_controls_dialog(parent, shortcuts=None, main_window=None):
    """Create a nonmodal editor without registering application shortcuts."""
    return ControlsDialog(parent, shortcuts, main_window)
