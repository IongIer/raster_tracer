"""Settings validation must keep bindings usable and failed changes atomic."""

import tempfile
import unittest
from pathlib import Path

from qgis.PyQt.QtCore import QEvent, QSettings, Qt
from qgis.PyQt.QtGui import QKeyEvent

from ..shortcuts import DEFAULT_BINDINGS, SETTINGS_PREFIX, ShortcutSettings
from .utilities import get_qgis_app


class ShortcutSettingsTest(unittest.TestCase):
    def setUp(self):
        get_qgis_app()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = QSettings(
            str(Path(self.directory.name) / "keys.ini"), QSettings.Format.IniFormat
        )
        self.shortcuts = ShortcutSettings(settings=self.settings)

    def test_duplicate_and_multistroke_apply_leave_all_settings_unchanged(self):
        for value in ("B", "Ctrl+K, Ctrl+C", "Ctrl+", "NotAKey"):
            with self.subTest(value=value):
                bindings = dict(DEFAULT_BINDINGS, toggle_straight=value)
                with self.assertRaises(ValueError):
                    self.shortcuts.apply(bindings, "Alt")
                self.assertEqual(self.shortcuts.bindings, DEFAULT_BINDINGS)
                self.assertEqual(self.shortcuts.finish_modifier, "Shift")
                self.assertEqual(self.settings.allKeys(), [])

    def test_invalid_saved_settings_use_defaults_and_cleared_keys_remain_disabled(self):
        self.settings.setValue(SETTINGS_PREFIX + "toggle_straight", "garbage")
        self.settings.setValue(SETTINGS_PREFIX + "sample_color", "")
        self.settings.setValue(SETTINGS_PREFIX + "finish_modifier", "garbage")
        loaded = ShortcutSettings(settings=self.settings)
        self.assertEqual(loaded.bindings["toggle_straight"], "A")
        self.assertEqual(loaded.bindings["sample_color"], "")
        self.assertEqual(loaded.finish_modifier, "Shift")
        self.settings.setValue(SETTINGS_PREFIX + "toggle_straight", "D")
        self.assertEqual(
            ShortcutSettings(settings=self.settings).bindings, DEFAULT_BINDINGS
        )

    def test_exact_modifiers_and_disabled_mouse_override(self):
        self.shortcuts.apply(dict(DEFAULT_BINDINGS, toggle_straight="Ctrl+J"), "Alt")
        for modifiers, expected in (
            (Qt.KeyboardModifier.NoModifier, None),
            (Qt.KeyboardModifier.ControlModifier, "toggle_straight"),
            (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
                None,
            ),
        ):
            self.assertEqual(
                self.shortcuts.action(
                    QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_J, modifiers)
                ),
                expected,
            )
        self.assertTrue(self.shortcuts.finish_inverted(Qt.KeyboardModifier.AltModifier))
        self.assertFalse(
            self.shortcuts.finish_inverted(Qt.KeyboardModifier.ShiftModifier)
        )
        self.shortcuts.apply(DEFAULT_BINDINGS, "None")
        self.assertFalse(
            self.shortcuts.finish_inverted(
                Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ShiftModifier
            )
        )

    def test_new_enhanced_default_preserves_existing_custom_e_binding(self):
        self.settings.setValue(SETTINGS_PREFIX + "sample_color", "E")
        self.settings.setValue(SETTINGS_PREFIX + "undo_segment", "Ctrl+Z")
        loaded = ShortcutSettings(settings=self.settings)
        self.assertEqual(loaded.bindings["sample_color"], "E")
        self.assertEqual(loaded.bindings["undo_segment"], "Ctrl+Z")
        self.assertEqual(loaded.bindings["toggle_enhanced_tracing"], "")
        self.assertEqual(
            loaded.action(
                QKeyEvent(
                    QEvent.Type.KeyPress, Qt.Key.Key_E, Qt.KeyboardModifier.NoModifier
                )
            ),
            "sample_color",
        )
        loaded.apply(loaded.bindings, loaded.finish_modifier)
        self.assertEqual(
            ShortcutSettings(settings=self.settings).bindings, loaded.bindings
        )
