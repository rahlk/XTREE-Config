from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xPlan',
    version='0.0.1',
    description='Package for xPlan',
    long_description=readme,
    author='Rahul Krishna',
    author_email='rkrish11@ncsu.edu',
    url='https://github.com/rahlk/xPlan',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)


