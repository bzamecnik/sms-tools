"""
This regression test allows to fix the outputs once and then compare subsequent
outputs to the previous ones. This allows to spot an error during refactoring
quickly. Many examples of models and transformations are ran in this test.

# save expected outputs once to the expected/ dir
$ python regression_test.py --save-expected

# then run the test against the expected outputs
$ ./run_regression_test.sh

The actual outputs are stored in the actual/ dir to find out what was
different.

The test also makes a code coverage report.

$ open cover/index.html

The files are compared using the MD5 hash. Missing files are spotted, while
superfluous files are not checked.

The numpy random seed is fixed to allow replicability in stochastic models.
"""

from __future__ import print_function
import argparse
import glob
from hashlib import md5
import os
import shutil

import numpy as np


# matplotlib without a GUI
import matplotlib as mpl

mpl.use('Agg')

import smst.ui

# An ugly hack to use the correct directory with sounds even for smst installed
# as a package.
smst.ui.demo_sound_path = lambda file_name: os.path.join('../sounds', file_name)

# TODO: try to make the module handing a bit less verbose

from smst.ui.models import (
    dftModel_function,
    stftModel_function,
    harmonicModel_function,
    hprModel_function,
    hpsModel_function,
    sineModel_function,
    sprModel_function,
    spsModel_function,
    stochasticModel_function
)

from smst.ui.transformations import (
    harmonicTransformations_function,
    hpsMorph_function,
    hpsTransformations_function,
    sineTransformations_function,
    stftMorph_function,
    stochasticTransformations_function
)

modules = {
    'dftModel_function': dftModel_function,
    'stftModel_function': stftModel_function,
    'harmonicModel_function': harmonicModel_function,
    'hprModel_function': hprModel_function,
    'hpsModel_function': hpsModel_function,
    'sineModel_function': sineModel_function,
    'sprModel_function': sprModel_function,
    'spsModel_function': spsModel_function,
    'stochasticModel_function': stochasticModel_function,
    'harmonicTransformations_function': harmonicTransformations_function,
    'hpsMorph_function': hpsMorph_function,
    'hpsTransformations_function': hpsTransformations_function,
    'sineTransformations_function': sineTransformations_function,
    'stftMorph_function': stftMorph_function,
    'stochasticTransformations_function': stochasticTransformations_function
}


def test_generator():
    clean_dir('actual/')
    for module_name in modules:
        yield single_module, module_name


def single_module(module_name):
    print('testing module:', module_name)
    clean_output()
    run_module(modules[module_name])
    save_output(module_name, 'actual')
    assert_module_outputs(module_name)


def run_module(module):
    # fixed random seed to allow replicable results
    np.random.seed(0)
    module.main(interactive=False, plotFile=True)


def assert_module_outputs(module_name):
    exp_dir = 'expected/%s/' % module_name
    expected_files = glob.glob('%soutput*/*' % exp_dir)
    for exp_file in expected_files:
        if not exp_file.endswith('.png'):
            actual_file = exp_file.replace('expected/', 'actual/')
            assert_file_equals(exp_file, actual_file)


def assert_file_equals(file1, file2):
    assert file_md5(file1) == file_md5(file2)


def file_md5(file):
    return md5(open(file, 'rb').read()).hexdigest()


def clean_output():
    for output_dir in ['output_plots', 'output_sounds']:
        clean_dir(output_dir)


def clean_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)


def save_output(module_name, target):
    dir = '%s/%s' % (target, module_name)
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)
    shutil.move('output_plots/', dir)
    shutil.move('output_sounds/', dir)


def save_expected_outputs():
    for module_name, module in modules.iteritems():
        print('running module:', module_name)
        clean_output()
        run_module(module)
        save_output(module_name, 'expected')


def parse_args():
    parser = argparse.ArgumentParser(description='Regression test')
    parser.add_argument('--save-expected', action='store_true', default=False,
                        help='save expected outputs')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.save_expected:
        save_expected_outputs()
    else:
        for func, module_name in test_generator():
            func(module_name)
