from setuptools import setup, find_packages

setup_kwargs = {
        "name": "VitLib_PyTorch",
        "version": "0.1.0",
        "description": "",
        "author": "Kotetsu0000",
        'install_requires' : [
            'torch',
        ],
        'packages': find_packages(),
    }

setup(**setup_kwargs)
