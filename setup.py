import pathlib
from setuptools import setup, find_packages

#HERE = pathlib.Path(__file__).parent

VERSION = '0.1.2'
PACKAGE_NAME = 'rave'
AUTHOR = 'Yinuo Han'
AUTHOR_EMAIL = 'yh458@cam.ac.uk'
URL = 'https://github.com/yinuohan/Rave'

LICENSE = 'GNU AFFERO GENERAL PUBLIC LICENSE 3.0'
DESCRIPTION = 'Modelling the structure of edge-on debris disk.'
#LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'matplotlib',
      'astropy',
      'scipy',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      #long_description=LONG_DESCRIPTION,
      long_description=DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )