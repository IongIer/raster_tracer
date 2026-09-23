"""Run the suite against source or an extracted plugin ZIP in real QGIS."""

import argparse
import faulthandler
import importlib.util
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


def load_package(name, directory):
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-zip", type=Path)
    parser.add_argument("--expected-qgis")
    parser.add_argument("--expected-qt-major", type=int)
    parser.add_argument("--failfast", action="store_true")
    parser.add_argument("--core-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    faulthandler.enable()
    # A broken task or modal dialog must fail CI instead of hanging forever.
    faulthandler.dump_traceback_later(180, exit=True)
    with tempfile.TemporaryDirectory(prefix="raster-tracer-tests-") as temp:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ["XDG_CONFIG_HOME"] = str(Path(temp) / "config")
        os.environ["XDG_DATA_HOME"] = str(Path(temp) / "data")
        os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

        if not args.core_only:
            import numpy as np
            from qgis.core import Qgis
            from qgis.PyQt.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

            print(
                f"QGIS {Qgis.QGIS_VERSION}; Qt {QT_VERSION_STR}; "
                f"PyQt {PYQT_VERSION_STR}; NumPy {np.__version__}",
                flush=True,
            )
            if args.expected_qgis:
                major, minor = map(int, args.expected_qgis.split("."))
                if Qgis.QGIS_VERSION_INT // 100 != major * 100 + minor:
                    raise RuntimeError("Container has the wrong QGIS version")
            if args.expected_qt_major:
                if int(QT_VERSION_STR.split(".")[0]) != args.expected_qt_major:
                    raise RuntimeError("Container has the wrong Qt major version")

        plugin_root = root
        if args.plugin_zip:
            with zipfile.ZipFile(args.plugin_zip) as archive:
                archive.extractall(Path(temp) / "plugin")
            plugin_root = Path(temp) / "plugin" / "raster_tracer"
        load_package("raster_tracer", plugin_root)
        load_package("raster_tracer.test", root / "test")
        tests = [
            f"raster_tracer.test.{path.stem}"
            for path in sorted(
                (root / "test").glob("test_core.py" if args.core_only else "test_*.py")
            )
        ]
        suite = unittest.defaultTestLoader.loadTestsFromNames(tests)
        result = unittest.TextTestRunner(verbosity=2, failfast=args.failfast).run(suite)
        if not args.core_only:
            from raster_tracer.test.utilities import stop_qgis_widgets

            stop_qgis_widgets()
        faulthandler.cancel_dump_traceback_later()
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
