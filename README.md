# Spectral Modeling Synthesis Tools

[![PyPI version](https://img.shields.io/pypi/v/smst.svg)](https://pypi.python.org/pypi/smst)
[![Build Status](https://travis-ci.org/bzamecnik/sms-tools.svg?branch=master)](https://travis-ci.org/bzamecnik/sms-tools)
[![Read The Docs](https://readthedocs.org/projects/smst/badge/?version=latest)](https://smst.readthedocs.org/en/latest/?badge=latest)
[![Chat of Gitter](https://img.shields.io/gitter/room/bzamecnik/sms-tools.svg)](https://gitter.im/bzamecnik/sms-tools)
![Supported Python versions](https://img.shields.io/pypi/pyversions/smst.svg)
![License](https://img.shields.io/pypi/l/smst.svg)
![PyPI downloads/month](https://img.shields.io/pypi/dm/smst.svg)

[SMS Tools](http://mtg.upf.edu/technologies/sms) is a set of techniques and software implementations for the analysis, transformation and synthesis of musical sounds based on various spectral modeling approaches. These techniques can be used for synthesis, processing and coding applications, while some of the intermediate results might also be applied to other music related problems, such as sound source separation, musical acoustics, music perception, or performance analysis. The basic model and implementation was developed in the PhD thesis by X. Serra in 1989 and since then many extensions have been proposed at [MTG-UPF](http://mtg.upf.edu/) and by other researchers.

This repository contains a library written in Python (with a bit of C) and complementary lecture materials for the [Audio Signal Processing for Music Applications](https://www.coursera.org/course/audio) course.

## Forks

There is the official repository (MTG) and many forks. This one is maintained, heavily refactored, has better API, docs, etc. but it is not backwards compatible with the current ASPMA assignments and instructional videos. For general usage of the library it is recommended to use this fork.

- https://github.com/MTG/sms-tools.git - the official repository (may not be up-to-date)
- https://github.com/bzamecnik/sms-tools.git - a maintained fork (this repo)

## Project structure

- `smst` - the SMS tools Python package
  - `ui` - command-line and graphical interface for demo purposes
- `tests` - automated tests
- `lectures` - lecture slides + code to generate the plots and other resources
- `sounds` - selected example sounds from Freesound used in the course
- `workspace` - place for student assignments

## Requirements

- Python 2.7
- numpy - numerical computations
- scipy - other scientific computations
- matplotlib - plotting
- cython - for some parts written in C

Optional:

- ipython - interactive notebook
- [essentia](http://essentia.upf.edu/) - audio features extraction

## How to install?

Make sure you have Python installed (see below).

In case you just want to use the library, install it from PyPI. This automatically install its dependendies.

```
$pip install smst
```

In case you're taking the ASPMA course or would like to use the example sounds and scripts or you would like to develop and extend the algorithms, obtain the whole source repository and build the package yourself.

### The repository

If you have `git` installed, just clone the repository:

```
git clone https://github.com/bzamecnik/sms-tools.git
```

Otherwise [download the current version as a ZIP](https://github.com/bzamecnik/sms-tools/archive/master.zip) and extract it.

### Python & its packages

#### Anaconda

The easiest and free way to install a working Python environment is [Anaconda](https://www.continuum.io/downloads). It has most of the required dependendies already bundled. However, you can also try the packages provided by you platform's native package system (apt, brew, etc.).

```python
# make sure Anaconda is installed
# create a virtual environment (eg. named 'smstools')
# make sure to install matplotlib via conda before installing it via pip!
$ conda env create -n smstools python=2.7 matplotlib
$ source activate smstools
$ pip install smst
```

#### Ubuntu

In Ubuntu (which we strongly recommend) in order to install all these modules it is as simple as typing in the Terminal:

```
$ sudo apt-get install python-dev ipython python-numpy python-matplotlib python-scipy cython
```

#### Mac OS X

In OSX (which we do not support but that should work) you install these modules by typing in the Terminal:

```
$ pip install ipython numpy matplotlib scipy cython
```

#### Building & installing

SMS tools are provided in the `smst` Python package.

It is needed to build the cython extensions and also it is convenient to have
the library installed in the system, so that it can be easily imported without
relying on some absolute location.

##### Development mode

In case you plan to modify the library code you can install the `smst` package in the development mode, ie. all code changes will come into effect without the
need to reinstall the package.

```
sms-tools$ pip install -e .
```

##### Building the package

In case you plan just to use the package and not modify its code often you can
just build and install the package.

```
sms-tools$ pip install .
```

#### Uninstalling

```
sms-tools$ pip uninstall smst
```

## How to use?

The scripts to run the graphical user interface (GUI) are expected to run in
the project directory. The sound paths are relative to it.

- `sounds` - input sounds
- `output_sounds` - output sounds processed by the models

### Models GUI

The basic sound analysis/synthesis functions, or models, are in the directory `smst` and there is a graphical interface and individual example functions in `smst/ui/models`. To execute the models GUI type the following command. Note that is has been installed via pip.

```
sms-tools$ smst-ui-models
```

### Transformations GUI

To execute the transformations GUI that calls various sound transformation functions type:

```
sms-tools$ smst-ui-transformations
```

### Coding projects/assignments

To modify the existing code, or to create your own using some of the functions, we recommend to use the `workspace` directory. Typically you would copy a file from `smst/ui/models` or from `smst/ui/transformations` to that directory, modify the code, and execute it from there (you will have to change some of the paths inside the files).

Look at the many examples of usage of the library in `smst/ui/*` and in `lectures/*/plots-code`. Also do not be afraid to look at the library sources.

## Available models and transformations

The code can be imported as python modules.

### Models

```
audio -> [analysis] -> model -> [synthesis] -> reconstructed audio
```

A model provides a different representation of audio than the time-domain samples. The models live as modules in the `smst` package.

Each model typically has a `fromAudio()` method which performs analysis and a `toAudio()` method which performs synthesis.

- `dft` - [Discrete Fourier Transform](smst/models/dft.py) - spectrum of a single frame
- `stft` - [Short-time Fourier Transform](smst/models/stft.py) - spectrogram
- `sine` - [Sinusoidal model](smst/models/sine.py) - for plain tones
- `harmonic` - [Harmonic model](smst/models/harmonic.py) - for harmonic tones
- `stochastic` - [Stochastic model](smst/models/stochastic.py) - for noises
- `spr` - [Sinusoidal + residual model](smst/models/spr.py)
- `sps` - [Sinusoidal + stochastic model](smst/models/sps.py)
- `hpr` - [Harmonic + residual model](smst/models/hpr.py)
- `hps` - [Harmonic + stochastic model](smst/models/hps.py)

### Transformations

```
model -> [transformation] -> transformed model
```

Audio can be transformed by modifying its model. Each transformation belongs to a model. Thus also transformations can be found in the `smst` package.

- `stft` - [STFT transformations](smst/models/stft.py)
  - `filter()`
  - `morph()`
- `sine` - [Sinusoidal transformations](smst/models/sine.py)
  - `scaleTime()`
  - `scaleFrequencies()`
- `harmonic` - [Harmonic transformations](smst/models/harmonic.py)
  - `scaleFrequencies()`
- `stochastic` - [Stochastic transformations](smst/models/stochastic.py)
  - `scaleTime()`
- `hps` - [Harmonic + stochastic transformations](smst/models/hps.py)
  - `scaleTime()`
  - `morph()`

## Documentation

The current [documentation](http://smst.readthedocs.org/en/latest/) is hosted on ReadTheDocs.org.

### How to generate documentation?

In case you want to manually generate the documentation using Sphinx take the following steps.

Regenerate the API reference into RST (reStructuredText) files.

```
sms-tools$ cd docs/
docs$ make apidoc
```

Build the documentation from RST into HTML.

```
sms-tools$ python setup.py build_sphinx
sms-tools$ open build/html/index.html
```

## History of releases

[Changelog](CHANGELOG.md) - releases with their major changes.

## Authors

The [people](AUTHORS) who contributed to this software.

## License

All the software is distributed with the [Affero GPL license](http://www.gnu.org/licenses/agpl-3.0.en.html), the lecture slides are distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0) license and the sounds in this repository are released under [Creative Commons Attribution 4.0](http://creativecommons.org/licenses/by/4.0/) (CC BY 4.0) license.
