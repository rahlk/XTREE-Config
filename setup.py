from __future__ import print_function

from setuptools import setup

try:
    from pypandoc import convert

    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print('pandoc is not installed.')
    read_md = lambda f: open(f, 'r').read()
with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xPlan',
    version='0.0.1',
    description='Package for xPlan',
    long_description=read_md('README.rst'),
    author='Rahul Krishna',
    py_modules=['xPlan'],
    install_requires=[
        "networkx>=1.9.1",
        "gsq>=0.1.5",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "scipy",
        "click",
        "matplotlib",
        "pandas",
        "pypandoc"],
    author_email='rkrish11@ncsu.edu',
    url='https://github.com/rahlk/xPlan',
    license=license,
    )
