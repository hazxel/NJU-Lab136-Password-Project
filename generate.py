import torch
import logging
import os
import json

from data_prep import Password as P
from model import Generator
from config import *

from sys import argv
from getopt import getopt


logger = logging.getLogger()
logger.setLevel(logging.INFO)
output_path = DEFAULT_SAMPLE_PATH
gen_num = 200
print_pass = False

opts, args = getopt(argv[1:], "-h-p-l:-o:-n:", ["help", "logging=", "output_path=", "number="])

for opt_name, opt_value in opts:
    if opt_name in ('-h','--help'):
        print("Ooops, we don't have help info now :)")
    if opt_name in ('-p'):
        print_pass = True
    if opt_name in ('-l', '--logging'):
        if opt_value == "debug":
            logger.setLevel(logging.DEBUG)
    if opt_name in ('-o', '--output_path'):
        output_path = opt_value
    if opt_name in ('-n', '--number'):
        gen_num = int(opt_value)      
    
p = P()

logging.debug("Loading generator...")
checkpoint = torch.load(DEFAULT_CHECKPOINT_PATH, map_location = device)
g = Generator(GEN_HIDDEN_SIZE, GEN_NEURON_SIZE).to(device)
g.load_state_dict(checkpoint['gen_state_dict'])

logging.debug("Generating passwords...")
gen_pass = g.generate_N(p, n_generate = gen_num)
logging.debug("Removing duplicate passwords...")
gen_pass = set(gen_pass)

if print_pass:
    logging.debug("{} passwords in total:".format(len(gen_pass)))
    print(gen_pass)
else:
    logging.debug("Saving passwords to json file...")
    with open(output_path, 'w',encoding='utf-8') as file:
        json.dump(gen_pass, file, ensure_ascii = True)
        
    logging.info("Done.")
