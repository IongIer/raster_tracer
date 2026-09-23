"""Keep pyrcc5 output usable with the Qt binding selected by QGIS."""

import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
path.write_text(
    source.replace("from PyQt5 import QtCore", "from qgis.PyQt import QtCore"),
    encoding="utf-8",
)
