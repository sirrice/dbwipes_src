#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()
from setuptools import setup, find_packages
import dbtruck

setup(name="scorpion",
      version=dbtruck.__version__,
      description="Outlier Explanation",
      license="MIT",
      author="Eugene Wu",
      author_email="eugenewu@mit.edu",
      url="http://github.com/sirrice/dbwipes_src",
      include_package_data = True,      
      packages = find_packages(),
      package_dir = {'dbtruck' : 'dbtruck'},
      scripts = ['server.py'],
      package_data = { 'dbtruck' : ['data/*'] },
      install_requires = [
        'argparse', 'DateUtils', 
			  'pyquery', 'flask',
        "sqlalchemy", "psycopg2", 'matplotlib',
        'pyparsing', 'rtree', 'scikit-learn',
        'orange'],
      keywords= "")
