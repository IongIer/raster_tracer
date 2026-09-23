"""Build an installable ZIP from the working tree, including every module."""

import argparse
import configparser
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(root / 'pb_tool.cfg')
    metadata = configparser.ConfigParser()
    metadata.read(root / 'metadata.txt')
    plugin_name = config['plugin']['name']
    version = metadata['general']['version']
    output = args.output or root / 'dist' / f'{plugin_name}-{version}.zip'
    files = config['files']
    names = set(files['python_files'].split() + files['extras'].split())
    names.add(files['main_dialog'])
    names.update(str(path.relative_to(root))
                 for path in (root / 'i18n').glob('*.qm'))
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(root / name, f'{plugin_name}/{name}')
    print(output)


if __name__ == '__main__':
    main()
