import os
import random
import logging
import numpy as np
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["OMP_NUM_THREADS"] = '1'

def setup_logger(log_dir=None, log_name="train.log"):
    logger = logging.getLogger("IGCN-BERT")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log(msg):
    print(msg, flush=True)