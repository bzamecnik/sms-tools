"""SMS tools installation script.

Packaging docs: https://packaging.python.org/en/latest/distributing.html
setup.py adapted from https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# the the cython extension
from setuptools import Extension
from Cython.Distutils import build_ext
import numpy as np

# To use a consistent encoding
from codecs import open

from os import path
import re

# here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

# Cython extensions
sourcefiles = [
    'smst/utils/utilFunctions_C/utilFunctions.c',
    'smst/utils/utilFunctions_C/cutilFunctions.pyx'
]
ext_modules = [Extension(
    'smst.utils.utilFunctions_C.utilFunctions_C',
    sourcefiles,
    libraries=['m'],
    # include_dirs=py_inc + np_inc
    include_dirs=[np.get_include()]
)]

def read_version():
    """
    Reads the project version without importing the module.
    This prevent importing a wrong module and importing the module before its
    dependencies are built.
    """
    version = ''
    with open('smst/__init__.py', 'r') as file:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            file.read(),
            re.MULTILINE
        ).group(1)
    if not version:
        raise RuntimeError('Could not find package version from __init__.py')
    return version

setup(
    name='smst',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=read_version(),

    description='SMS tools - spectral audio modeling/synthesis/transformations',
    # long_description=long_description,

    # The project's main homepage.
    url='https://github.com/MTG/sms-tools',

    # Author details
    # See the AUTHORS file for the complete list of authors
    author='Xavier Serra, Fabia Serra Arrizabalaga, Sankalp Gulati and others',
    author_email='xavier.serra@upf.edu',

    # Choose your license
    license='AGPL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',

        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
    ],

    # What does your project relate to?
    keywords='audio spectral modelling analysis synthesis',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy', 'matplotlib', 'cython'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     '': ['AUTHORS', 'LICENSE', 'README.md']
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'smst-ui-models=smst.ui.models.models_GUI:main',
            'smst-ui-transformations=smst.ui.transformations.transformations_GUI:main',
            'smst-model=smst.ui:main_model',
            'smst-transformation=smst.ui:main_transformation',
        ],
    },

    # Cython extensions
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules

)
