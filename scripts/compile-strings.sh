#!/bin/bash
set -euo pipefail
LRELEASE=${1:-lrelease}
LOCALES=${2:-}

cd "$(dirname "$0")/.."

for LOCALE in ${LOCALES}
do
    echo "Processing: ${LOCALE}.ts"
    "$LRELEASE" "i18n/${LOCALE}.ts"
done
