import os 
import os.path as osp
import logging 
import time 

def Logger(log_dir, name=None):
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s")

    localtime = time.strftime("%Y_%m_%d_%H_%M_%S")
    if name is None:
        logfile = osp.join(log_dir,localtime+".log")
        fh = logging.FileHandler(logfile,mode="w")

    else:
        logfile = osp.join(log_dir,name+".log")
        fh = logging.FileHandler(logfile,mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 