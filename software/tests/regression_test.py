"""
This regression test allows to fix the outputs once and then compare subsequent
outputs to the previous ones. This allows to spot an error during refactoring
quickly. Many examples of models and transfomations are ran in this test.

# save expected outputs once to the expected/ dir
$ python regression_test.py --save-expected

# then run the test against the expected outputs
$ sh run_regression_test.sh

The actual outputs are stored in the actual/ dir to find out what was
different.

The files are compared using the MD5 hash. Missing files are spotted, while
superfluous files are not checked.

The numpy random seed is fixed to allow replicability in stochastic models.
"""

from __future__ import print_function
import argparse
import glob
from hashlib import md5
import numpy as np
import os
import shutil
import sys

# matplotlib without a GUI
import matplotlib as mpl
mpl.use('Agg')

local_packages = ['../models_interface/', '../transformations_interface/']
for p in local_packages:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), p))

module_names = ['%s_function' % m for m in
        ['%sModel' % model for model in [
            'dft', 'harmonic', 'hpr', 'hps', 'sine', 'spr', 'sps', 'stochastic']]
        + ['harmonicTransformations', 'hpsMorph', 'sineTransformations',
            'stftMorph', 'stochasticTransformations']]

modules = {m:__import__(m) for m in module_names}

def test_generator():
    clean_dir('actual/')
    for module_name in module_names:
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
