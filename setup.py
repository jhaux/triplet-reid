from setuptools import setup

setup(name='triplet_reid',
      version='1.0',
      description='triple-reid made pip-installable',
      url='https://github.com/VisualComputingInstitute/triplet-reid',
      author='Hermans*, Alexander and Beyer*, Lucas and Leibe, Bastian',
      packages=['triplet_reid'],
      install_requires=[
          'tensorflow',
          ],
      zip_safe=False)
