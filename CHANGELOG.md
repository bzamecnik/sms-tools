# Changelog

## Next release

## 0.3.0 (2015-12-30)

- [#3](https://github.com/bzamecnik/sms-tools/issues/3) - code refactoring
  - the DFT model completely refactored to be more readable
  - API methods renamed to more descriptive names and underscore-lowercase
    - this breaks compatibility with the previous version
  - classes in the GUI renamed
  - [#36](https://github.com/bzamecnik/sms-tools/issues/36) utility module was split into submodules based on the usage
- [#10](https://github.com/bzamecnik/sms-tools/issues/10) code mostly formatted according to PEP8, imports optimized
- documentation build using Sphinx and published on [ReadTheDocs.org](https://smst.readthedocs.org/)

## 0.2.0 (2015-12-28)

- simplified API
  - [#31](https://github.com/bzamecnik/sms-tools/issues/31) - packages/modules
  - [#33](https://github.com/bzamecnik/sms-tools/issues/33) - function names
- automatic build & test via Travis CI
- optimized imports
- fixed some issues after refactoring in lecture plot code

## 0.1.1 (2015-12-26)

- bugs fixed:
  - [#29](https://github.com/bzamecnik/sms-tools/issues/29) Missing utilFunctions.h in the source package.

## 0.1.0 (2015-12-26)

- structured into Python modules
- packaged into a package called `smst`
- [published via PyPI](https://pypi.python.org/pypi/smst)
- can be installed via `pip install smst`
- no need to building the cython extension manually
- commands for launching the UI for models and transformations easily
- code refactoring
- regression test

## Older history

The code was not versioned before. Look at the [commit history](https://github.com/MTG/sms-tools/commits/master).
