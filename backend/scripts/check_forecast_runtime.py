"""Read-only dependency audit. Run with the intended dedicated Python interpreter.

Does not import ML frameworks, install packages, load weights, train or write data.
"""
import importlib.metadata
import importlib.util
import json
import sys

PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'torch': 'torch', 'lightgbm': 'lightgbm',
            'holidays': 'holidays', 'sklearn': 'scikit-learn', 'pyarrow': 'pyarrow'}


def audit():
    packages = {}
    for module, distribution in PACKAGES.items():
        installed = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if installed else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[distribution] = {'module_available': installed, 'version': version}
    return {'python': sys.version.split()[0], 'executable': sys.executable, 'packages': packages,
        'imports_available': all(p['module_available'] for p in packages.values()),
        'model_inference_verified': False, 'training_package_versions_verified': False,
        'note': 'Module discovery alone does not prove binary compatibility or model parity.'}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
