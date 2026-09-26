"""Edit a finished feature and extend its layer without opening a table."""

from qgis.core import Qgis, QgsField
from qgis.gui import QgsAttributeEditorContext, QgsAttributeForm, QgsEditorWidgetWrapper
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QCoreApplication, QMetaType, Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def field_creation_unavailable(layer):
    """Explain why a layer cannot accept a field through its edit buffer."""
    if sip.isdeleted(layer) or not layer.isValid():
        return FinishedFeatureDialog.tr("The layer is no longer available.")
    if not layer.isEditable():
        return FinishedFeatureDialog.tr("Enable layer editing before adding a field.")
    if layer.isEditCommandActive():
        return FinishedFeatureDialog.tr(
            "Finish the current layer edit before adding a field."
        )
    provider = layer.dataProvider()
    if not provider or not (
        provider.capabilities() & Qgis.VectorProviderCapability.AddAttributes
    ):
        return FinishedFeatureDialog.tr("This layer does not support adding fields.")
    if not provider.nativeTypes():
        return FinishedFeatureDialog.tr(
            "This layer does not offer any supported field types."
        )
    return ""


def field_validation_error(layer, field):
    """Validate common naming rules and the provider's native type limits."""
    reason = field_creation_unavailable(layer)
    if reason:
        return reason
    name = field.name()
    if not name or name != name.strip():
        return FinishedFeatureDialog.tr(
            "Enter a field name without leading or trailing spaces."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return FinishedFeatureDialog.tr(
            "Field names cannot contain control characters."
        )
    if name.casefold() in {field.name().casefold() for field in layer.fields()}:
        return FinishedFeatureDialog.tr("A field with this name already exists.")
    # The edit buffer accepts long names which the Shapefile driver would
    # silently truncate at commit, potentially producing duplicate names.
    provider = layer.dataProvider()
    if "shapefile" in provider.storageType().lower():
        encoding = provider.encoding() or "UTF-8"
        try:
            encoded_name = name.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return FinishedFeatureDialog.tr(
                "The field name cannot be stored in this layer's encoding."
            )
        if len(encoded_name) > 10:
            return FinishedFeatureDialog.tr(
                "Shapefile field names must fit within 10 bytes."
            )
    if not provider.supportedType(field):
        return FinishedFeatureDialog.tr(
            "This field type, length, or precision is not supported by the layer."
        )
    if field.length() > 0 and field.precision() > field.length():
        return FinishedFeatureDialog.tr("Precision cannot be greater than length.")
    return ""


class AddFieldDialog(QDialog):
    """Select a field definition using the layer provider's supported types."""

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        self._layer_available = True
        self.setWindowTitle(FinishedFeatureDialog.tr("Add field to layer"))
        layout = QVBoxLayout(self)
        explanation = QLabel(
            FinishedFeatureDialog.tr(
                "The new field applies to every feature in this layer. "
                "It remains if you cancel the feature form."
            ),
            self,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        fields = QFormLayout()
        layout.addLayout(fields)
        self.name_edit = QLineEdit(self)
        self.type_combo = QComboBox(self)
        self.length_spin = QSpinBox(self)
        self.precision_spin = QSpinBox(self)
        fields.addRow(FinishedFeatureDialog.tr("Name"), self.name_edit)
        fields.addRow(FinishedFeatureDialog.tr("Type"), self.type_combo)
        fields.addRow(FinishedFeatureDialog.tr("Length"), self.length_spin)
        fields.addRow(FinishedFeatureDialog.tr("Precision"), self.precision_spin)
        self.native_types = layer.dataProvider().nativeTypes()
        for native_type in self.native_types:
            self.type_combo.addItem(native_type.mTypeDesc)
        self.error_label = QLabel(self)
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layer.editingStopped.connect(self._layer_unavailable)
        layer.willBeDeleted.connect(self._layer_unavailable)
        self.type_combo.currentIndexChanged.connect(self._update_type)
        self.name_edit.textChanged.connect(self._validate)
        self.length_spin.valueChanged.connect(self._validate)
        self.precision_spin.valueChanged.connect(self._validate)
        self._update_type()
        self.name_edit.setFocus()

    def _layer_unavailable(self):
        if self._layer_available:
            self._layer_available = False
            self.reject()

    def _update_type(self):
        index = self.type_combo.currentIndex()
        if index < 0:
            self._validate()
            return
        native_type = self.native_types[index]
        for widget, minimum, maximum in (
            (self.length_spin, native_type.mMinLen, native_type.mMaxLen),
            (self.precision_spin, native_type.mMinPrec, native_type.mMaxPrec),
        ):
            widget.setEnabled(maximum > 0)
            widget.setRange(max(0, minimum), max(0, maximum))
            widget.setValue(max(0, minimum))
        self._validate()

    def field(self):
        index = self.type_combo.currentIndex()
        if index < 0:
            return None
        native_type = self.native_types[index]
        return QgsField(
            self.name_edit.text(),
            QMetaType.Type(native_type.mType),
            native_type.mTypeName,
            self.length_spin.value(),
            self.precision_spin.value(),
            "",
            QMetaType.Type(native_type.mSubType),
        )

    def _validate(self):
        field = self.field()
        error = (
            field_validation_error(self.layer, field)
            if field is not None
            else FinishedFeatureDialog.tr(
                "This layer does not offer any supported field types."
            )
        )
        self.error_label.setText(error)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not error)
        return not error

    def accept(self):
        if self._validate():
            super().accept()


class FinishedFeatureDialog(QDialog):
    """A native QGIS form whose Cancel leaves the completed geometry intact."""

    @staticmethod
    def tr(message):
        # Qualified .tr calls let pylupdate5 extract the runtime context.
        return QCoreApplication.translate("FinishedFeatureDialog", message)

    def __init__(self, iface, layer, feature_id, parent=None):
        super().__init__(parent if parent is not None else iface.mainWindow())
        self.iface = iface
        self.layer = layer
        self.feature_id = feature_id
        self.form = None
        self._layer_available = True
        self.setWindowTitle(
            FinishedFeatureDialog.tr("Finished line attributes — {layer}").format(
                layer=layer.name()
            )
        )
        self.layout = QVBoxLayout(self)
        self.empty_label = QLabel(
            FinishedFeatureDialog.tr("This layer has no attribute fields yet."), self
        )
        self.layout.addWidget(self.empty_label)
        self.custom_form_label = QLabel(
            FinishedFeatureDialog.tr(
                "This layer uses a custom form. If a new field is missing, "
                "add it to the layout in Form settings."
            ),
            self,
        )
        self.custom_form_label.setWordWrap(True)
        self.layout.addWidget(self.custom_form_label)
        actions = QHBoxLayout()
        self.add_field_button = QPushButton(
            FinishedFeatureDialog.tr("Add field to layer…"), self
        )
        self.add_field_button.setAutoDefault(False)
        self.add_field_button.clicked.connect(self._choose_field)
        actions.addWidget(self.add_field_button)
        self.form_settings_button = QPushButton(
            FinishedFeatureDialog.tr("Form settings…"), self
        )
        self.form_settings_button.setAutoDefault(False)
        self.form_settings_button.clicked.connect(self._configure_form)
        actions.addWidget(self.form_settings_button)
        actions.addStretch()
        self.layout.addLayout(actions)
        self.error_label = QLabel(self)
        self.error_label.setWordWrap(True)
        self.layout.addWidget(self.error_label)
        self._build_form()
        layer.editingStopped.connect(self._layer_unavailable)
        layer.willBeDeleted.connect(self._layer_unavailable)
        self.resize(500, 400)

    def _focus_field(self, name=None):
        wrappers = self.form.findChildren(QgsEditorWidgetWrapper)
        for wrapper in wrappers:
            if name is not None and wrapper.field().name() != name:
                continue
            widget = wrapper.widget()
            if not widget.isEnabled() or widget.isHidden():
                continue
            if widget.focusPolicy() & Qt.FocusPolicy.TabFocus:
                widget.setFocus(Qt.FocusReason.OtherFocusReason)
                return
            for child in widget.findChildren(QWidget):
                if (
                    child.isEnabled()
                    and not child.isHidden()
                    and child.focusPolicy() & Qt.FocusPolicy.TabFocus
                ):
                    child.setFocus(Qt.FocusReason.OtherFocusReason)
                    return
        self.add_field_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def _layer_unavailable(self):
        if not self._layer_available:
            return
        self._layer_available = False
        for dialog in self.findChildren(AddFieldDialog):
            dialog._layer_unavailable()
        self._remove_form()
        self.reject()

    def _form_values(self):
        feature = self.form.currentFormFeature()
        return {field.name(): feature[field.name()] for field in feature.fields()}

    def _remove_form(self):
        if self.form is not None:
            self.layout.removeWidget(self.form)
            # Delete now so its schema-change signals and Python form hooks do
            # not run during addAttribute or a nested layer-properties dialog.
            sip.delete(self.form)
            self.form = None

    def _build_form(self, values=None):
        if not self._layer_available or sip.isdeleted(self.layer):
            return
        context = QgsAttributeEditorContext()
        context.setFormMode(QgsAttributeEditorContext.FormMode.StandaloneDialog)
        context.setMapCanvas(self.iface.mapCanvas())
        if hasattr(self.iface, "vectorLayerTools"):
            context.setVectorLayerTools(self.iface.vectorLayerTools())
        self.form = QgsAttributeForm(
            self.layer, self.layer.getFeature(self.feature_id), context, self
        )
        # hideButtonBox()/Embed enable auto-save on layer modification checks.
        # Keep the native buttons and constraint state, taking over acceptance
        # so an unsuccessful native save never closes this dialog.
        self.form.showButtonBox()
        self.form.disconnectButtonBox()
        buttons = self.form.findChild(QDialogButtonBox, "buttonBox")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.form)
        for name, value in (values or {}).items():
            if self.layer.fields().indexFromName(name) >= 0:
                self.form.changeAttribute(name, value)
        self.empty_label.setVisible(len(self.layer.fields()) == 0)
        custom = (
            self.layer.editFormConfig().layout()
            != Qgis.AttributeFormLayout.AutoGenerated
        )
        self.custom_form_label.setVisible(custom)
        self.form_settings_button.setVisible(custom)
        reason = field_creation_unavailable(self.layer)
        self.add_field_button.setEnabled(not reason)
        self.add_field_button.setToolTip(reason)
        self._focus_field()

    def _choose_field(self):
        dialog = AddFieldDialog(self.layer, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.add_field(dialog.field())
        finally:
            dialog.deleteLater()

    def add_field(self, field):
        """Add one independent schema edit, preserving unaccepted widget values."""
        error = field_validation_error(self.layer, field)
        self.error_label.setText(error)
        if error:
            return False
        values = self._form_values()
        self._remove_form()
        self.layer.beginEditCommand(FinishedFeatureDialog.tr("Add field to layer"))
        try:
            added = self.layer.addAttribute(field)
            if added:
                self.layer.endEditCommand()
            else:
                self.layer.destroyEditCommand()
                self.error_label.setText(
                    FinishedFeatureDialog.tr("The layer could not add this field.")
                )
        except Exception:
            self.layer.destroyEditCommand()
            raise
        finally:
            self._build_form(values)
        if added and self.form is not None:
            self._focus_field(field.name())
        return added

    def _configure_form(self):
        values = self._form_values()
        self._remove_form()
        try:
            self.iface.showLayerProperties(self.layer, "mOptsPage_AttributesForm")
        finally:
            self._build_form(values)

    def accept(self):
        if not self._layer_available or sip.isdeleted(self.layer):
            self.reject()
            return
        if not self.layer.getFeature(self.feature_id).isValid():
            self.error_label.setText(
                FinishedFeatureDialog.tr("The finished feature is no longer available.")
            )
            return
        if self.form.save():
            super().accept()


def show_finished_feature_form(iface, layer, feature_id):
    """Open a modal form for an existing line and return whether it was accepted."""
    if sip.isdeleted(layer) or not layer.getFeature(feature_id).isValid():
        return False
    dialog = FinishedFeatureDialog(iface, layer, feature_id)
    try:
        return dialog.exec() == QDialog.DialogCode.Accepted
    finally:
        # Dispose native form connections before later undo, save, or layer
        # removal events can reach a hidden dialog.
        sip.delete(dialog)
