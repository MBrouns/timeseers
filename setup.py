import os

from setuptools import setup, find_packages

base_packages = ["numpy>=1.15.4", "scipy>=1.2.0", "pandas>=0.23.4", "pymc3"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='seers',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=base_packages,
    description='An hierarchical version of Facebooks prophet in PyMC3',
    author='Matthijs Brouns',
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
)
