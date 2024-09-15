from setuptools import setup, find_packages

setup_kwargs = {
        "name": "VitLib_PyTorch",
        "version": "0.2.4",
        "description": "",
        "author": "Kotetsu0000",
        'install_requires' : [
            'torch',
            'opencv-python',
            'einops',
            'timm',
        ],
        'packages': find_packages(),
    }

setup(**setup_kwargs)
