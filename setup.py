from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='data_flywheel',
    version='0.0.1',
    description='Iteratively correct your bounding box annotations using Tesla\'s data engine framework.',
    author='Dickson Neoh',
    url="https://github.com/dnth/data-flywheel",
    packages=find_packages(),
    install_requires=required,
)