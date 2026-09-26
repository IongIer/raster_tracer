"""Real native forms preserve drafts, schema changes, and independent undo."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qgis.core import (
    NULL,
    Qgis,
    QgsAttributeEditorField,
    QgsFeature,
    QgsField,
    QgsFieldConstraints,
    QgsGeometry,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QMetaType, QTimer
from qgis.PyQt.QtTest import QTest
from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox

from ..attribute_form import (
    AddFieldDialog,
    FinishedFeatureDialog,
    field_validation_error,
    show_finished_feature_form,
)
from .utilities import get_qgis_app


class AttributeFormTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.canvas, cls.iface, cls.window = get_qgis_app()

    def setUp(self):
        self.layer = QgsVectorLayer(
            "MultiLineString?crs=EPSG:3857&field=notes:string",
            "Finished lines",
            "memory",
        )
        self.assertTrue(self.layer.startEditing())
        self.feature = QgsFeature(self.layer.fields())
        self.feature.setGeometry(QgsGeometry.fromWkt("MULTILINESTRING((0 0,1 1))"))
        self.feature["notes"] = "original"
        self.layer.beginEditCommand("Finished line")
        self.assertTrue(self.layer.addFeature(self.feature))
        self.layer.endEditCommand()
        self.dialogs = []

    def tearDown(self):
        for dialog in reversed(self.dialogs):
            if not sip.isdeleted(dialog):
                sip.delete(dialog)
        if not sip.isdeleted(self.layer):
            self.layer.rollBack()
            sip.delete(self.layer)

    def dialog(self):
        dialog = FinishedFeatureDialog(self.iface, self.layer, self.feature.id())
        self.dialogs.append(dialog)
        return dialog

    def decimal_field(self, name="elevation"):
        return QgsField(name, QMetaType.Type.Double, "double")

    def test_cancel_preserves_finished_line_and_discards_widget_edits(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "not accepted")
        self.assertEqual(dialog.form.currentFormFeature()["notes"], "not accepted")
        # A modification check must not invoke QgsAttributeForm auto-save.
        self.assertTrue(self.layer.isModified())
        dialog.reject()
        self.assertEqual(self.layer.getFeature(self.feature.id())["notes"], "original")
        self.assertEqual(self.layer.featureCount(), 1)
        self.assertEqual(self.layer.undoStack().count(), 1)

    def test_add_field_preserves_unsaved_values_then_cancel_keeps_only_schema(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "still being typed")
        self.assertTrue(dialog.add_field(self.decimal_field()))
        self.assertEqual(dialog.form.currentFormFeature()["notes"], "still being typed")
        dialog.form.changeAttribute("elevation", 123.5)
        self.assertTrue(self.layer.isModified())
        self.assertEqual(self.layer.getFeature(self.feature.id())["notes"], "original")
        self.assertEqual(self.layer.getFeature(self.feature.id())["elevation"], NULL)
        dialog.reject()
        self.assertEqual(self.layer.fields().names(), ["notes", "elevation"])
        self.assertEqual(self.layer.undoStack().count(), 2)
        sip.delete(dialog)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.fields().names(), ["notes"])
        self.assertEqual(self.layer.featureCount(), 1)
        self.layer.undoStack().redo()
        self.assertEqual(self.layer.fields().names(), ["notes", "elevation"])

    def test_accepted_attributes_schema_and_line_have_separate_undo(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "accepted")
        self.assertTrue(dialog.add_field(self.decimal_field()))
        dialog.form.changeAttribute("elevation", 123.5)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(
            self.layer.getFeature(self.feature.id()).attributes(), ["accepted", 123.5]
        )
        self.assertTrue(self.layer.isEditable())
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)
        sip.delete(dialog)
        stack = self.layer.undoStack()
        self.assertEqual(stack.count(), 3)
        stack.undo()
        self.assertEqual(
            self.layer.getFeature(self.feature.id()).attributes(), ["original", NULL]
        )
        stack.undo()
        self.assertEqual(self.layer.fields().names(), ["notes"])
        self.assertEqual(self.layer.featureCount(), 1)
        stack.undo()
        self.assertEqual(self.layer.featureCount(), 0)
        for _ in range(3):
            stack.redo()
        self.assertEqual(
            self.layer.getFeature(self.feature.id()).attributes(), ["accepted", 123.5]
        )

    def test_empty_layer_can_add_and_edit_first_field(self):
        self.assertTrue(self.layer.deleteAttribute(0))
        dialog = self.dialog()
        self.assertFalse(dialog.empty_label.isHidden())
        self.assertTrue(dialog.add_field(self.decimal_field("depth")))
        self.assertTrue(dialog.empty_label.isHidden())
        dialog.form.changeAttribute("depth", -3.25)
        dialog.accept()
        self.assertEqual(self.layer.getFeature(self.feature.id())["depth"], -3.25)

    def test_empty_and_empty_custom_forms_have_visible_dismiss_buttons(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                if custom:
                    config = self.layer.editFormConfig()
                    config.setLayout(Qgis.AttributeFormLayout.DragAndDrop)
                    config.invisibleRootContainer().clear()
                    self.layer.setEditFormConfig(config)
                else:
                    self.assertTrue(self.layer.deleteAttribute(0))
                dialog = self.dialog()
                dialog.show()
                self.app.processEvents()
                buttons = dialog.form.findChild(QDialogButtonBox, "buttonBox")
                cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
                self.assertTrue(cancel.isVisible())
                self.assertTrue(cancel.isEnabled())
                cancel.click()
                self.assertFalse(dialog.isVisible())

    def test_duplicate_and_invalid_fields_leave_input_and_undo_unchanged(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "draft")
        for name in ("notes", "NOTES", "", " name ", "new\nfield"):
            with self.subTest(name=name):
                self.assertFalse(dialog.add_field(self.decimal_field(name)))
                self.assertEqual(self.layer.fields().names(), ["notes"])
                self.assertEqual(dialog.form.currentFormFeature()["notes"], "draft")
                self.assertEqual(self.layer.undoStack().count(), 1)

    def test_add_failure_does_not_discard_values_or_create_undo_entry(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "draft")
        with patch.object(self.layer, "addAttribute", return_value=False):
            self.assertFalse(dialog.add_field(self.decimal_field()))
        self.assertEqual(dialog.form.currentFormFeature()["notes"], "draft")
        self.assertEqual(self.layer.fields().names(), ["notes"])
        self.assertEqual(self.layer.undoStack().count(), 1)
        self.assertFalse(self.layer.isEditCommandActive())

    def test_field_creation_disabled_when_provider_does_not_support_it(self):
        provider = self.layer.dataProvider()
        capabilities = (
            provider.capabilities() & ~Qgis.VectorProviderCapability.AddAttributes
        )
        with patch.object(provider, "capabilities", return_value=capabilities):
            dialog = self.dialog()
            self.assertFalse(dialog.add_field_button.isEnabled())
            self.assertIn("does not support", dialog.add_field_button.toolTip())
            self.assertFalse(dialog.add_field(self.decimal_field()))
        self.assertEqual(self.layer.fields().names(), ["notes"])

    def test_custom_form_is_retained_and_settings_preserve_unsaved_input(self):
        config = self.layer.editFormConfig()
        config.setLayout(Qgis.AttributeFormLayout.DragAndDrop)
        root = config.invisibleRootContainer()
        root.clear()
        root.addChildElement(QgsAttributeEditorField("notes", 0, root))
        self.layer.setEditFormConfig(config)
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "draft")
        self.assertTrue(dialog.add_field(self.decimal_field()))
        self.assertEqual(dialog.form.currentFormFeature()["notes"], "draft")
        self.assertFalse(dialog.custom_form_label.isHidden())
        self.assertFalse(dialog.form_settings_button.isHidden())
        with patch.object(
            self.iface, "showLayerProperties", create=True
        ) as show_properties:
            dialog._configure_form()
        show_properties.assert_called_once_with(self.layer, "mOptsPage_AttributesForm")
        self.assertEqual(dialog.form.currentFormFeature()["notes"], "draft")
        config = self.layer.editFormConfig()
        self.assertEqual(config.layout(), Qgis.AttributeFormLayout.DragAndDrop)
        self.assertEqual(
            [child.name() for child in config.invisibleRootContainer().children()],
            ["notes"],
        )

    def test_failed_native_save_keeps_dialog_open(self):
        dialog = self.dialog()
        finished = Mock()
        dialog.finished.connect(finished)
        with patch.object(dialog.form, "save", return_value=False):
            dialog.accept()
        finished.assert_not_called()

    def test_hard_constraint_still_blocks_native_ok_after_adding_field(self):
        self.layer.setFieldConstraint(
            0,
            QgsFieldConstraints.Constraint.ConstraintNotNull,
            QgsFieldConstraints.ConstraintStrength.ConstraintStrengthHard,
        )
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", NULL)
        self.assertTrue(dialog.add_field(self.decimal_field()))
        buttons = dialog.form.findChild(QDialogButtonBox, "buttonBox")
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        self.assertEqual(self.layer.getFeature(self.feature.id())["notes"], "original")
        dialog.form.changeAttribute("notes", "valid")
        self.assertTrue(ok.isEnabled())
        ok.click()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.layer.getFeature(self.feature.id())["notes"], "valid")

    def test_initial_and_added_fields_receive_typing_focus(self):
        dialog = self.dialog()
        dialog.show()
        self.app.processEvents()
        self.assertIsNotNone(dialog.focusWidget())
        self.assertTrue(dialog.form.isAncestorOf(dialog.focusWidget()))
        self.assertTrue(dialog.add_field(self.decimal_field()))
        self.app.processEvents()
        self.assertTrue(dialog.form.isAncestorOf(dialog.focusWidget()))
        QTest.keyClicks(dialog.focusWidget(), "27.5")
        dialog.accept()
        self.assertEqual(self.layer.getFeature(self.feature.id())["elevation"], 27.5)

    def test_schema_creation_does_not_nest_an_external_edit_command(self):
        dialog = self.dialog()
        self.layer.beginEditCommand("Another edit")
        try:
            self.assertFalse(dialog.add_field(self.decimal_field()))
            self.assertTrue(self.layer.isEditCommandActive())
            self.assertEqual(self.layer.fields().names(), ["notes"])
        finally:
            self.layer.destroyEditCommand()

    def test_stopping_edits_closes_form_without_saving_widget_values(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "not accepted")
        finished = Mock()
        dialog.finished.connect(finished)
        self.assertTrue(self.layer.commitChanges())
        finished.assert_called_once_with(QDialog.DialogCode.Rejected)
        self.assertIsNone(dialog.form)
        self.assertEqual(next(self.layer.getFeatures())["notes"], "original")

    def test_layer_deletion_during_form_settings_closes_without_rebuilding(self):
        dialog = self.dialog()
        finished = Mock()
        dialog.finished.connect(finished)
        with patch.object(
            self.iface,
            "showLayerProperties",
            side_effect=lambda *_: sip.delete(self.layer),
            create=True,
        ):
            dialog._configure_form()
        finished.assert_called_once_with(QDialog.DialogCode.Rejected)
        self.assertIsNone(dialog.form)

    def test_layer_deletion_with_live_native_form_does_not_leave_callbacks(self):
        dialog = self.dialog()
        dialog.form.changeAttribute("notes", "not accepted")
        finished = Mock()
        dialog.finished.connect(finished)
        sip.delete(self.layer)
        finished.assert_called_once_with(QDialog.DialogCode.Rejected)
        self.assertIsNone(dialog.form)

    def test_layer_deletion_rejects_nested_field_dialog_before_feature_dialog(self):
        dialog = self.dialog()
        closed = []
        dialog.finished.connect(lambda _: closed.append("feature"))

        def remove_layer():
            child = dialog.findChild(AddFieldDialog)
            child.finished.connect(lambda _: closed.append("field"))
            sip.delete(self.layer)

        QTimer.singleShot(0, remove_layer)
        dialog._choose_field()
        self.assertEqual(closed, ["field", "feature"])
        self.assertIsNone(dialog.form)

    def test_field_dialog_offers_native_types_and_validates_duplicates(self):
        dialog = AddFieldDialog(self.layer)
        self.dialogs.append(dialog)
        self.assertEqual(
            dialog.type_combo.count(), len(self.layer.dataProvider().nativeTypes())
        )
        ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.assertFalse(ok.isEnabled())
        dialog.name_edit.setText("notes")
        self.assertFalse(ok.isEnabled())
        dialog.name_edit.setText("measurement")
        for index, native_type in enumerate(dialog.native_types):
            # The memory provider also advertises its custom geometry QVariant
            # type; choosing it must not crash on either Qt major version.
            with self.subTest(type=native_type.mTypeName):
                dialog.type_combo.setCurrentIndex(index)
                field = dialog.field()
                self.assertEqual(field.name(), "measurement")
                self.assertEqual(field.typeName(), native_type.mTypeName)
                self.assertEqual(
                    ok.isEnabled(), not bool(field_validation_error(self.layer, field))
                )

    def test_shapefile_name_limit_is_checked_before_schema_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lines.shp"
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "ESRI Shapefile"
            result = QgsVectorFileWriter.writeAsVectorFormatV3(
                self.layer,
                str(path),
                QgsProject.instance().transformContext(),
                options,
            )
            self.assertEqual(result[0], QgsVectorFileWriter.NoError, result)
            layer = QgsVectorLayer(str(path), "Shapefile", "ogr")
            try:
                self.assertTrue(layer.startEditing())
                error = field_validation_error(
                    layer,
                    QgsField("long_field_name", QMetaType.Type.QString, "String", 30),
                )
                self.assertIn("10 bytes", error)
                self.assertEqual(len(layer.fields()), 1)
                self.assertEqual(layer.undoStack().count(), 0)
            finally:
                layer.rollBack()
                sip.delete(layer)

    def test_missing_feature_does_not_open_a_dialog(self):
        with patch("raster_scribe.attribute_form.FinishedFeatureDialog") as constructor:
            self.assertFalse(show_finished_feature_form(self.iface, self.layer, 999))
        constructor.assert_not_called()
