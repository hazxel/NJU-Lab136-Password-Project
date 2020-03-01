import logging
import os
import json
from sys import argv
from getopt import getopt
from config import *
from data_prep import Password

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sample_path = DEFAULT_SAMPLE_PATH
test_path = DEFAULT_TEST_PATH
output_path = DEFAULT_EVAL_PATH

opts, args = getopt(argv[1:], "-h-l:-o:-s:-t:", ["help", "logging=", "output_path=", "sample_path=", "test_path"])

for opt_name, opt_value in opts:
    if opt_name in ('-h','--help'):
        print("Ooops, we don't have help info now, please check the code :)")
    if opt_name in ('-l', '--logging'):
        if opt_value == "debug":
            logger.setLevel(logging.DEBUG)
    if opt_name in ('-s', '--sample_path'):
        sample_path = opt_value
    if opt_name in ('-t', '--test_path'):
        test_path = opt_value
    if opt_name in ('-o', '--output_path'):
        output_path = opt_value


assert os.path.isfile(sample_path)
logging.debug("Loading generated passwords...")
with open(sample_path, 'r', encoding = 'utf-8') as samples:
    gen_pass = json.load(samples)
gen_pass = set(gen_pass)

assert os.path.isfile(test_path)
real_pass = open(test_path, encoding='utf-8', errors='ignore').read().strip().split('\n')
real_pass = [Password.unicodeToAscii(password) for password in real_pass]
real_pass = set(real_pass)

in_test = len(gen_pass.intersection(real_pass))
print("{} passwords generated\n".format(len(gen_pass)))
print("{}%% generated password are in test set\n".format(in_test * 100.0 / len(gen_pass)))
print("{}%% password in test set are covered\n".format(in_test * 100.0 / len(real_pass)))