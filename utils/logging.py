import os
import json
from torch.utils.tensorboard import SummaryWriter

def save_config(config,save_path):
    os.makedirs(save_path,exist_ok=True)
    with open(os.path.join(save_path,"config.json"),"w") as f:
        json.dump(config,f,indent=4)

def create_writer(log_dir):
    os.makedirs(log_dir,exist_ok=True)
    return SummaryWriter(log_dir=log_dir)