# coding=utf-8
"""Resources test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = "mkondratyev85@gmail.com"
__date__ = "2019-11-09"
__copyright__ = "Copyright 2019, Mikhail Kondratyev"

import unittest

from qgis.PyQt.QtGui import QIcon

from .. import resources  # noqa: F401 - Registers the plugin's Qt resources.
from .utilities import get_qgis_app

QGIS_APP = get_qgis_app()


class RasterScribeResourcesTest(unittest.TestCase):
    """Check the plugin's registered Qt resources."""

    def test_icon_png(self):
        """The plugin icon loads and renders from its resource path."""
        path = ":/plugins/raster_scribe/icon.png"
        icon = QIcon(path)
        self.assertFalse(icon.isNull())
        self.assertFalse(icon.pixmap(24, 24).isNull())


if __name__ == "__main__":
    unittest.main()
