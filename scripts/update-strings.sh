#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "No locales requested; translation files are unchanged."
  exit 0
fi

cd "$(dirname "$0")/.."
mkdir -p i18n
PYLUPDATE=${PYLUPDATE:-pylupdate5}

# Use the runtime manifest so environments, tests and builds are never scanned.
# Generated resources.py contains no translatable source strings.
SOURCE_FILES=$("${PYTHON:-python3}" - <<'PY'
import configparser

config = configparser.ConfigParser()
with open("pb_tool.cfg") as manifest:
    config.read_file(manifest)
files = config["files"]
names = set(files["python_files"].split())
names.update(files["main_dialog"].split())
names.discard("resources.py")
print("\n".join(sorted(names)))
PY
)
mapfile -t PYTHON_FILES <<< "$SOURCE_FILES"

# Get newest source timestamp so we don't update strings unnecessarily.
CHANGED_FILES=0
for PYTHON_FILE in "${PYTHON_FILES[@]}"; do
  CHANGED=$(stat -c %Y "$PYTHON_FILE")
  if [ "$CHANGED" -gt "$CHANGED_FILES" ]; then
    CHANGED_FILES=$CHANGED
  fi
done

UPDATE=false
for LOCALE in "$@"; do
  TRANSLATION_FILE="i18n/$LOCALE.ts"
  if [ ! -f "$TRANSLATION_FILE" ]; then
    UPDATE=true
    break
  fi

  MODIFICATION_TIME=$(stat -c %Y "$TRANSLATION_FILE")
  if [ "$CHANGED_FILES" -gt "$MODIFICATION_TIME" ]; then
    UPDATE=true
    break
  fi
done

if [ "$UPDATE" = true ]; then
  printf '%s\n' "${PYTHON_FILES[@]}"
  echo "Please provide translations by editing the translation files below:"
  for LOCALE in "$@"; do
    echo "i18n/$LOCALE.ts"
    "$PYLUPDATE" -noobsolete "${PYTHON_FILES[@]}" -ts "i18n/$LOCALE.ts"
  done
else
  echo "No source files have changed since the last translation update."
fi
