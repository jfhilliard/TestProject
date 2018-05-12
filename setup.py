# -*- coding: utf-8 -*-

from distutils.core import setup


setup(name='TestProject',
      version='1.0',
      description='My Python Sandbox',
      author='Jonathan Hilliard',
      author_email='jfhilliard@gmail.com',
      packages=['Sandbox'],
      package_dir={'Sandbox': 'Sandbox/playground'}
      )
