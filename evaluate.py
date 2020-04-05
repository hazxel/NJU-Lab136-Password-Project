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
training_path = DEFAULT_TRAINING_PATH

opts, args = getopt(argv[1:], "-h-l:-s:-tr:-te:", ["help", "logging=", "sample_path=", "training_path=" "test_path="])

for opt_name, opt_value in opts:
    if opt_name in ('-h','--help'):
        print("Ooops, we don't have help info now, please check the code :)")
    if opt_name in ('-l', '--logging'):
        if opt_value == "debug":
            logger.setLevel(logging.DEBUG)
    if opt_name in ('-s', '--sample_path'):
        sample_path = opt_value
    if opt_name in ('-te', '--test_path'):
        test_path = opt_value
    if opt_name in ('-tr', '--training_path'):
        training_path = opt_value


assert os.path.isfile(sample_path)
logging.debug("Loading generated passwords...")
with open(sample_path, 'r', encoding = 'utf-8') as samples:
    gen_pass = json.load(samples)
gen_pass = set(gen_pass)

assert os.path.isfile(test_path)
logging.debug("Loading test set...")
with open(test_path, 'r', encoding = 'utf-8') as test_set:
	real_pass = json.load(test_set)
#real_pass = open(test_path, encoding='utf-8', errors='ignore').read().strip().split('\n')
#real_pass = [Password.unicodeToAscii(password) for password in real_pass]
real_pass = set(real_pass)

assert os.path.isfile(training_path)
logging.debug("Loading training set...")
with open(training_path, 'r', encoding = 'utf-8') as training_set:
	training_pass = json.load(training_set)
training_pass = set(training_pass)

in_test = gen_pass.intersection(real_pass)
in_training = gen_pass.intersection(training_pass)
intersection = in_test.intersection(in_training)

print("  Among {} passwords generated:".format(len(gen_pass)))
print("  - {} passwords are in test set({} in total)".format(len(in_test), len(real_pass)))
print("  - {} passwords are in training set({} in total)".format(len(in_training), len(training_pass)))
print("  - {}% generated password are in test set and not in training set".format((len(in_test) - len(intersection)) * 100.0 / len(gen_pass)))
print("  - {}% password in test set are covered".format(len(in_test) * 100.0 / len(real_pass)))


