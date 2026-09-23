# coding=utf-8
"""Safe Translations Test.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

from .utilities import get_qgis_app

__author__ = "ismailsunni@yahoo.co.id"
__date__ = "12/10/2011"
__copyright__ = "Copyright 2012, Australia Indonesia Facility for Disaster Reduction"
import os
import unittest

from qgis.PyQt.QtCore import QCoreApplication, QTranslator

QGIS_APP = get_qgis_app()


class SafeTranslationsTest(unittest.TestCase):
    """Test translations work."""

    def test_qgis_translations(self):
        """Test that translations work."""
        file_path = os.path.join(os.path.dirname(__file__), "af.qm")
        if not os.path.isfile(file_path):
            self.skipTest("Optional Afrikaans translation fixture is not built")
        translator = QTranslator()
        self.assertTrue(translator.load(file_path))
        QCoreApplication.installTranslator(translator)
        self.addCleanup(QCoreApplication.removeTranslator, translator)

        expected_message = "Goeie more"
        real_message = QCoreApplication.translate("@default", "Good morning")
        self.assertEqual(real_message, expected_message)


if __name__ == "__main__":
    unittest.main()
