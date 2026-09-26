"""Conflict warnings reflect live, customized Qt and QGIS shortcuts."""

import unittest

from qgis.gui import QgsShortcutsManager
from qgis.PyQt import sip
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QKeySequence
from qgis.PyQt.QtWidgets import QAction, QMainWindow, QShortcut, QWidget

from ..shortcut_conflicts import find_shortcut_conflicts
from .utilities import get_qgis_app


class ShortcutConflictsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_qgis_app()

    def setUp(self):
        self.window = QMainWindow()
        self.other_window = QWidget()
        self.manager = QgsShortcutsManager(
            self.window, "/raster-scribe-test-shortcuts/"
        )

    def tearDown(self):
        sip.delete(self.window)
        sip.delete(self.other_window)

    def conflicts(self, bindings):
        return find_shortcut_conflicts(bindings, self.window, self.manager)

    def test_current_action_assignments_and_alternate_keys(self):
        action = QAction("&Select && inspect", self.window)
        action.setShortcuts([QKeySequence("A"), QKeySequence("Ctrl+J")])
        self.window.addAction(action)
        result = self.conflicts({"straight": "A", "sample": "Ctrl+J"})
        self.assertEqual(result["straight"][0].owner, "Select & inspect")
        self.assertFalse(result["straight"][0].partial)
        self.assertEqual(result["sample"][0].shortcut, "Ctrl+J")
        action.setShortcut(QKeySequence("Shift+A"))
        result = self.conflicts({"straight": "A", "sample": "Shift+A"})
        self.assertEqual(result["straight"], ())
        self.assertEqual(len(result["sample"]), 1)
        self.assertEqual(action.shortcut().toString(), "Shift+A")

    def test_registered_action_outside_main_window_and_deduplication(self):
        detached = QAction("Detached test action", self.other_window)
        self.manager.registerAction(detached, "A")
        attached = QAction("Attached test action", self.window)
        self.manager.registerAction(attached, "A")
        result = self.conflicts({"straight": "A"})["straight"]
        self.assertEqual(
            [conflict.owner for conflict in result],
            ["Attached test action", "Detached test action"],
        )
        self.manager.unregisterAction(detached)
        self.manager.unregisterAction(attached)

    def test_registered_shortcut_and_unregistered_descendant(self):
        registered = QShortcut(self.other_window)
        registered.setObjectName("Registered test shortcut")
        self.manager.registerShortcut(registered, "Ctrl+J")
        child = QWidget(self.window)
        unregistered = QShortcut(QKeySequence("A"), child)
        unregistered.setObjectName("Local test shortcut")
        result = self.conflicts({"straight": "A", "sample": "Ctrl+J"})
        self.assertEqual(result["straight"][0].owner, "Local test shortcut")
        self.assertEqual(result["sample"][0].owner, "Registered test shortcut")
        registered.setKey(QKeySequence("Ctrl+K"))
        self.assertEqual(self.conflicts({"sample": "Ctrl+J"})["sample"], ())
        self.manager.unregisterShortcut(registered)

    def test_first_stroke_conflicts_but_later_strokes_do_not(self):
        action = QAction("Multi-step test action", self.window)
        action.setShortcut(QKeySequence("Ctrl+K, A"))
        result = self.conflicts({"prefix": "Ctrl+K", "later": "A", "unmodified": "K"})
        self.assertTrue(result["prefix"][0].partial)
        self.assertEqual(result["prefix"][0].shortcut, "Ctrl+K, A")
        self.assertEqual(result["later"], ())
        self.assertEqual(result["unmodified"], ())

    def test_shifted_symbols_warn_for_qgis_unshifted_symbol_assignments(self):
        action = QAction("Symbol shortcut", self.window)
        for binding, qgis_binding in (
            ("Shift++", "+"),
            ("Ctrl+Shift++", "Ctrl++"),
            ("Alt+Shift+?", "Alt+?"),
            ("Shift+!", "!"),
        ):
            with self.subTest(binding=binding, qgis_binding=qgis_binding):
                action.setShortcut(QKeySequence(qgis_binding))
                result = self.conflicts({"symbol": binding})["symbol"]
                self.assertEqual(len(result), 1)
                self.assertFalse(result[0].partial)
                self.assertEqual(result[0].shortcut, qgis_binding)
        action.setShortcut(QKeySequence("Ctrl++, A"))
        result = self.conflicts({"symbol": "Ctrl+Shift++"})["symbol"]
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].partial)

    def test_symbol_alias_preserves_other_modifiers_and_shifted_letters(self):
        action = QAction("Distinct combinations", self.window)
        for binding, qgis_binding in (
            ("Ctrl+Shift++", "+"),
            ("Shift++", "Ctrl++"),
            ("+", "Shift++"),
            ("Shift+A", "A"),
            ("A", "Shift+A"),
            ("Ctrl+Shift+A", "Ctrl+A, D"),
            ("Shift+F1", "F1"),
            ("Shift+1", "1"),
        ):
            with self.subTest(binding=binding, qgis_binding=qgis_binding):
                action.setShortcut(QKeySequence(qgis_binding))
                self.assertEqual(self.conflicts({"test": binding})["test"], ())

    def test_potential_conflicts_include_disabled_and_widget_scoped_keys(self):
        action = QAction("Temporarily disabled", self.window)
        action.setShortcut(QKeySequence("A"))
        action.setEnabled(False)
        shortcut = QShortcut(QKeySequence("D"), self.window)
        shortcut.setObjectName("Focused-widget shortcut")
        shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        result = self.conflicts({"straight": "A", "dense": "D"})
        self.assertFalse(result["straight"][0].enabled)
        self.assertTrue(result["dense"][0].enabled)

    def test_empty_bindings_and_empty_qgis_sequences_do_not_conflict(self):
        action = QAction("No shortcut", self.window)
        action.setShortcut(QKeySequence())
        shortcut = QShortcut(self.window)
        shortcut.setObjectName("Empty shortcut")
        self.assertEqual(
            self.conflicts({"disabled": "", "unassigned": QKeySequence("A")}),
            {"disabled": (), "unassigned": ()},
        )

    def test_qshortcut_alternate_keys_when_supported(self):
        shortcut = QShortcut(QKeySequence("A"), self.window)
        shortcut.setObjectName("Shortcut alternatives")
        if hasattr(shortcut, "setKeys"):
            shortcut.setKeys([QKeySequence("A"), QKeySequence("D")])
            expected = 1
        else:
            expected = 0
        result = self.conflicts({"straight": "A", "dense": "D"})
        self.assertEqual(len(result["straight"]), 1)
        self.assertEqual(len(result["dense"]), expected)
