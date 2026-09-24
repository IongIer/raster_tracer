# coding=utf-8
"""DockWidget test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = "mkondratyev85@gmail.com"
__date__ = "2019-11-09"
__copyright__ = "Copyright 2019, Mikhail Kondratyev"

import unittest

from qgis.PyQt.QtWidgets import QDockWidget

from ..raster_scribe_dockwidget import RasterScribeDockWidget
from .utilities import get_qgis_app

QGIS_APP = get_qgis_app()


class RasterScribeDockWidgetTest(unittest.TestCase):
    """Check that the dock loads with the expected defaults."""

    def setUp(self):
        self.dockwidget = RasterScribeDockWidget(None)

    def tearDown(self):
        self.dockwidget.deleteLater()
        self.dockwidget = None

    def test_preview_defaults(self):
        """The dock starts with previews enabled and the default line width."""
        self.assertIsInstance(self.dockwidget, QDockWidget)
        self.assertTrue(self.dockwidget.checkBoxPreview.isChecked())
        self.assertAlmostEqual(self.dockwidget.previewWidthSpinBox.value(), 2.7)


if __name__ == "__main__":
    unittest.main()
