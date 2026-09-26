"""Persistent, tool-local keyboard bindings and the finish-form gesture."""

from qgis.PyQt.QtCore import QCoreApplication, QObject, QSettings, Qt, pyqtSignal
from qgis.PyQt.QtGui import QKeySequence


class RasterScribeShortcuts:
    @staticmethod
    def tr(message):
        return QCoreApplication.translate("RasterScribeShortcuts", message)


DEFAULT_BINDINGS = {
    "toggle_straight": "A",
    "toggle_dense_straight": "D",
    "sample_color": "T",
    "toggle_color_snap": "N",
    "undo_segment": "B",
    "cancel_pending": "Esc",
}
FINISH_MODIFIERS = ("Shift", "Ctrl", "Alt", "Meta", "None")
MODIFIER_FLAGS = {
    "Shift": Qt.KeyboardModifier.ShiftModifier,
    "Ctrl": Qt.KeyboardModifier.ControlModifier,
    "Alt": Qt.KeyboardModifier.AltModifier,
    "Meta": Qt.KeyboardModifier.MetaModifier,
    "None": Qt.KeyboardModifier.NoModifier,
}
SETTINGS_PREFIX = "RasterScribe/shortcuts/"


def enum_value(value):
    return int(getattr(value, "value", value))


def first_combination(sequence):
    """Qt5 exposes an integer; Qt6 exposes a QKeyCombination."""
    first = sequence[0]
    return first.toCombined() if hasattr(first, "toCombined") else int(first)


def sequence_label(portable):
    return QKeySequence(portable, QKeySequence.SequenceFormat.PortableText).toString(
        QKeySequence.SequenceFormat.NativeText
    )


def modifier_label(portable):
    if portable == "None":
        return RasterScribeShortcuts.tr("Disabled")
    # NativeText uses Command/Control symbols correctly on macOS.
    return (
        QKeySequence(enum_value(MODIFIER_FLAGS[portable]))
        .toString(QKeySequence.SequenceFormat.NativeText)
        .rstrip("+")
    )


def action_descriptions():
    return {
        "toggle_straight": RasterScribeShortcuts.tr("Toggle straight-line mode."),
        "toggle_dense_straight": RasterScribeShortcuts.tr(
            "Toggle straight-line mode with extra vertices."
        ),
        "sample_color": RasterScribeShortcuts.tr(
            "Sample a color under the pointer and return to tracing."
        ),
        "toggle_color_snap": RasterScribeShortcuts.tr("Toggle color snapping."),
        "undo_segment": RasterScribeShortcuts.tr(
            "Cancel pending work, undo the last tracing step, or remove the starting point."
        ),
        "cancel_pending": RasterScribeShortcuts.tr("Cancel the pending segment."),
    }


def normalize_binding(binding):
    if isinstance(binding, QKeySequence):
        sequence = binding
    elif isinstance(binding, str):
        sequence = QKeySequence(binding, QKeySequence.SequenceFormat.PortableText)
    else:
        raise ValueError(
            RasterScribeShortcuts.tr("Enter a key combination or clear the shortcut.")
        )
    if sequence.isEmpty():
        if isinstance(binding, str) and binding:
            raise ValueError(
                RasterScribeShortcuts.tr("The shortcut is not a valid key combination.")
            )
        return ""
    if sequence.count() != 1:
        raise ValueError(
            RasterScribeShortcuts.tr("Use one key combination, not a sequence of keys.")
        )
    combination = first_combination(sequence)
    key = combination & ~enum_value(Qt.KeyboardModifier.KeyboardModifierMask)
    if key in (
        0,
        enum_value(Qt.Key.Key_unknown),
        enum_value(Qt.Key.Key_Shift),
        enum_value(Qt.Key.Key_Control),
        enum_value(Qt.Key.Key_Alt),
        enum_value(Qt.Key.Key_Meta),
        enum_value(Qt.Key.Key_AltGr),
    ):
        raise ValueError(
            RasterScribeShortcuts.tr("Choose a key, optionally with modifiers.")
        )
    return sequence.toString(QKeySequence.SequenceFormat.PortableText)


def validation_errors(bindings, finish_modifier):
    errors, assigned = {}, {}
    for action in DEFAULT_BINDINGS:
        try:
            normalized = normalize_binding(bindings.get(action, ""))
        except ValueError as error:
            errors[action] = str(error)
            continue
        if normalized:
            if normalized in assigned:
                other = assigned[normalized]
                message = RasterScribeShortcuts.tr(
                    "This shortcut is assigned to another tracing action."
                )
                errors[action] = errors[other] = message
            assigned[normalized] = action
    if finish_modifier not in FINISH_MODIFIERS:
        errors["finish_modifier"] = RasterScribeShortcuts.tr(
            "Choose a finish-form modifier from the list."
        )
    return errors


class ShortcutSettings(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings()
        bindings = {}
        for action, default in DEFAULT_BINDINGS.items():
            try:
                bindings[action] = normalize_binding(
                    self.settings.value(SETTINGS_PREFIX + action, default)
                )
            except ValueError:
                bindings[action] = default
        # Malformed or duplicate saved bindings must not make an action ambiguous.
        self.bindings = (
            dict(DEFAULT_BINDINGS) if validation_errors(bindings, "Shift") else bindings
        )
        modifier = self.settings.value(SETTINGS_PREFIX + "finish_modifier", "Shift")
        self.finish_modifier = modifier if modifier in FINISH_MODIFIERS else "Shift"
        self._update_lookup()

    def _update_lookup(self):
        self._lookup = {
            first_combination(
                QKeySequence(binding, QKeySequence.SequenceFormat.PortableText)
            ): action
            for action, binding in self.bindings.items()
            if binding
        }

    def apply(self, bindings, finish_modifier):
        errors = validation_errors(bindings, finish_modifier)
        if errors:
            raise ValueError("\n".join(dict.fromkeys(errors.values())))
        normalized = {
            action: normalize_binding(bindings.get(action, ""))
            for action in DEFAULT_BINDINGS
        }
        self.bindings = normalized
        self.finish_modifier = finish_modifier
        self._update_lookup()
        for action, binding in normalized.items():
            self.settings.setValue(SETTINGS_PREFIX + action, binding)
        self.settings.setValue(SETTINGS_PREFIX + "finish_modifier", finish_modifier)
        self.changed.emit()

    def action(self, event, include_repeats=False):
        if event.isAutoRepeat() and not include_repeats:
            return None
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        return self._lookup.get(enum_value(event.key()) | enum_value(modifiers))

    def finish_inverted(self, modifiers):
        flag = MODIFIER_FLAGS[self.finish_modifier]
        return flag != Qt.KeyboardModifier.NoModifier and bool(modifiers & flag)
