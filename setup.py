import os

from setuptools import setup, find_packages

base_packages = ["numpy>=1.15.4", "pandas>=0.23.4", "pymc>=4.0"]
plot_packages = ["matplotlib>=3.2.1"]
dev_packages = ["pytest==5.3.4", "flake8>=3.7.9", "hypothesis==5.8.3"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='timeseers',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=base_packages,
    extras_require={
      "dev": dev_packages,
      "plot": plot_packages
    },
    description='An hierarchical version of Facebooks prophet in PyMC3',
    author='Matthijs Brouns',
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
)
