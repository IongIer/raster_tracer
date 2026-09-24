#/***************************************************************************
# RasterScribe
#
# This plugin traces the underlying raster map
#							 -------------------
#		begin				: 2019-11-09
#		git sha				: $Format:%H$
#		copyright			: (C) 2019 by Mikhail Kondratyev
#		email				: mkondratyev85@gmail.com
# ***************************************************************************/
#
#/***************************************************************************
# *																		 *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 2 of the License, or	 *
# *   (at your option) any later version.								   *
# *																		 *
# ***************************************************************************/

# Space-separated language codes for transup and transcompile.
LOCALES ?=
LRELEASE ?= lrelease
PYTHON ?= python3
PYRCC ?= pyrcc5
UV ?= uv

COMPILED_RESOURCE_FILES = resources.py
RESOURCE_SRC=$(shell grep '^ *<file' resources.qrc | sed 's@</file>@@g;s/.*>//g' | tr '\n' ' ')

.PHONY: default compile test package transup transcompile transclean clean doc lint format format-check
default:
	@echo "make lint / format / format-check: check or format Python with uv and Ruff"
	@echo "make package: build the plugin ZIP using checked-in resources.py"
	@echo "make compile: regenerate resources.py after asset changes (requires pyrcc5)"
	@echo "make test: run tests with a configured PyQGIS Python"
	@echo "See documentation/development.md and test/README.md for setup and tests."

compile: $(COMPILED_RESOURCE_FILES)

%.py : %.qrc $(RESOURCE_SRC)
	$(PYRCC) -o $*.py $<
	$(PYTHON) scripts/fix-resource-imports.py $*.py

%.qm : %.ts
	$(LRELEASE) $<

test:
	$(PYTHON) scripts/run-tests.py

package:
	$(PYTHON) scripts/package-plugin.py

transup:
	@PYTHON="$(PYTHON)" scripts/update-strings.sh $(LOCALES)

transcompile:
	@scripts/compile-strings.sh "$(LRELEASE)" "$(LOCALES)"

transclean:
	rm -f i18n/*.qm

clean:
	rm -f $(COMPILED_RESOURCE_FILES)

doc:
	@echo "Documentation is Markdown in documentation/; no build is needed."

lint:
	$(UV) run --locked ruff check .

format:
	$(UV) run --locked ruff format .

format-check:
	$(UV) run --locked ruff format --check .
