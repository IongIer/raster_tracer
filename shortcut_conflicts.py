"""Read live QGIS shortcut assignments without claiming or changing them."""

from dataclasses import dataclass
from unicodedata import category

from qgis.gui import QgsGui
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import QAction, QShortcut

from .shortcuts import enum_value, first_combination


@dataclass(frozen=True)
class ShortcutConflict:
    """A potential conflict; context and enabled state can change with focus."""

    owner: str
    shortcut: str
    partial: bool
    enabled: bool


def _owner_name(obj):
    """Remove menu mnemonics, preserving escaped literal ampersands."""
    if isinstance(obj, QAction) and obj.text():
        text = obj.text().split("\t", 1)[0]
        return text.replace("&&", "\0").replace("&", "").replace("\0", "&")
    if obj.objectName():
        return obj.objectName()
    parent = obj.parent()
    if parent is not None and parent.objectName():
        return parent.objectName()
    return obj.metaObject().className()


def _live_shortcuts(main_window, manager):
    objects = list(manager.listAll())
    if main_window is not None:
        objects.extend(main_window.findChildren(QAction))
        objects.extend(main_window.findChildren(QShortcut))
    seen = set()
    for obj in objects:
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            if isinstance(obj, QAction):
                sequences = obj.shortcuts()
            elif isinstance(obj, QShortcut):
                # Qt 6 supports alternate QShortcut keys; Qt 5 has one key.
                sequences = obj.keys() if hasattr(obj, "keys") else [obj.key()]
            else:
                continue
            owner, enabled = _owner_name(obj), obj.isEnabled()
            for sequence in sequences:
                if not sequence.isEmpty():
                    yield owner, sequence, enabled
        except RuntimeError:
            # A plugin can remove its actions while the controls are open.
            continue


def _potential_match(sequence, candidate):
    match = sequence.matches(candidate)
    if match != QKeySequence.SequenceMatch.NoMatch or sequence.count() != 1:
        return match
    combination = first_combination(sequence)
    shift = enum_value(Qt.KeyboardModifier.ShiftModifier)
    key = combination & ~enum_value(Qt.KeyboardModifier.KeyboardModifierMask)
    if (
        combination & shift
        and 0 <= key <= 0x10FFFF
        and category(chr(key))[0] in ("P", "S")
    ):
        # A native keymap can consume Shift when producing punctuation. For
        # example, the raw event Shift+Plus can activate QAction("+"). Qt's
        # QKeySequence.matches compares stored codes, without that keymap step.
        # Keep other modifiers and do not merge Shift+letter or function keys.
        return QKeySequence(combination & ~shift).matches(candidate)
    return match


def find_shortcut_conflicts(bindings, main_window, manager=None):
    """Return potential QGIS conflicts for each action's current binding.

    ``bindings`` maps tracing action IDs to QKeySequence objects or portable
    key-sequence strings. Values may be empty to disable an action. Results
    include exact matches and sequences whose first strokes match, so a
    single tracing key also flags a longer QGIS shortcut starting with it.
    Symbolic keys also check the unshifted symbol assignment, because native
    layouts can consume Shift when producing a symbol such as plus.

    Inspect the QGIS manager and unregistered window descendants on every
    call, including alternate and user-customized assignments. Keep disabled
    and widget-scoped shortcuts as potential conflicts: their availability
    can change when the user returns to the canvas. This is a read-only
    inventory of Qt shortcuts, not of operating-system or raw event handlers.
    """
    if manager is None:
        manager = QgsGui.shortcutsManager()
    live = tuple(_live_shortcuts(main_window, manager))
    result = {}
    for action, binding in bindings.items():
        sequence = (
            binding
            if isinstance(binding, QKeySequence)
            else QKeySequence.fromString(
                binding, QKeySequence.SequenceFormat.PortableText
            )
        )
        conflicts = []
        if not sequence.isEmpty():
            for owner, candidate, enabled in live:
                match = _potential_match(sequence, candidate)
                if match != QKeySequence.SequenceMatch.NoMatch:
                    conflict = ShortcutConflict(
                        owner=owner,
                        shortcut=candidate.toString(
                            QKeySequence.SequenceFormat.NativeText
                        ),
                        partial=match == QKeySequence.SequenceMatch.PartialMatch,
                        enabled=enabled,
                    )
                    if conflict not in conflicts:
                        conflicts.append(conflict)
        result[action] = tuple(
            sorted(conflicts, key=lambda conflict: (conflict.owner, conflict.shortcut))
        )
    return result
