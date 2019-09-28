import logging
import logging.handlers
import psutil
import os
import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Results summary')
parser.add_argument('results_dir', type=str, help='directory of results')
parser.add_argument('pid', type=int)
args = parser.parse_args()

results_dir = args.results_dir

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.handlers.TimedRotatingFileHandler(os.path.join(results_dir, 'memory_consumption_by_time.csv'), when='D', interval=2)
formatter = logging.Formatter('%(asctime)s, %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def get_memory_consumption():
    process = psutil.Process(args.pid)
    logger.info('{}'.format(process.memory_info()[0]))

try:
    while True:
        get_memory_consumption()
        time.sleep(60)
except KeyboardInterrupt as e:
    logging.shutdown()
    sys.exit()