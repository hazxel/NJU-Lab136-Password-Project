from sys import argv
from getopt import getopt

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sample_path = DEFAULT_SAMPLE_PATH
output_path = DEFAULT_EVAL_PATH

opts, args = getopt(argv[1:], "-h-l:-o:-i:", ["help", "logging=", "output_path=", "input_path="])

for opt_name, opt_value in opts:
    if opt_name in ('-h','--help'):
        print("Ooops, we don't have help info now :)")
    if opt_name in ('-l', '--logging'):
        if opt_value == "debug":
            logger.setLevel(logging.DEBUG)
    if opt_name in ('-i', '--input_path'):
        sample_path = opt_value
    if opt_name in ('-o', '--output_path'):
        output_path = opt_value

        
#len(list(set(generate_list[0:1000]).intersection(set(lines))))


if os.path.isfile(json_file_path) and not(update_all):
    logging.debug("Loading generated passwords...")
    with open(sample_path, 'r', encoding = 'utf-8') as samples:
        gen_pass = json.load(samples)