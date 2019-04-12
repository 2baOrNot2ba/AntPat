#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name = 'AntPat',
      version = '0.1',
      description = 'Antenna pattern tools.',
      author = 'Tobia D. Carozzi',
      author_email = 'tobia.carozzi@chalmers.se',
      packages = find_packages(exclude=('tests', 'docs')),
      license = 'ISC',
      classifiers = [
          'Development Status :: 1 - Planning',
          'Intended Audience :: Telecommunications Industry',
          'License :: OSI Approved :: ISC License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Communications :: Ham Radio',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization'
      ],
      install_requires=[
          'numpy>=1.10',
          'scipy>=0.16',
          'matplotlib'
      ],
      scripts = ['scripts/viewFFpat.py', 'scripts/viewJonespat.py']
     )
