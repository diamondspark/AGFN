import yaml
from easydict import EasyDict
import argparse
from utils.maplight import *
from iterators.samp_iter_finetune import build_train_loader
from finetuneSrc.ft_driver import train
from denovo_trainer import DockingFineTuner
from torch import nn as nn
import os
import torch
from rdkit import RDLogger, Chem
import torch.multiprocessing as mp
import torch.distributed as dist
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
RDLogger.DisableLog("rdApp.*")
import wandb
wandb.login()


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12388'
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    hps = config.finetuning
    default_hps= {"Z_learning_rate": 1e-3,}
    hps.update(default_hps)

    if hps.training_batch_size>16:
        if torch.cuda.device_count()>1:
            rank = 1 
        else:
            raise RuntimeError('Batch size too large for 1 GPU setup. Reduce training_batch_size in config.yml')
    if hps.task == 'QedxSaxDock':
        conditional_range_dict = {
                    'tpsa':[[10,200],[10,200], 0], 
                    'num_rings':[[1,5],[1,5],1], 
                    'sas': [[1,5],[1,5], 0], 
                    'qed':[[0.5,1],[0,1], 0],
                    }

    cond_prop_var= {
                'tpsa':20,
                'num_rings':1,
                'sas':1,
                'qed':1,
                }


    hps.task_conditionals= False
    hps.task_rewards_only = False

    hps.update(conditional_range_dict)

    wandb_project_name = hps.get("wandb_project_name", "GFN_Finetune") #"GFN_leadoptim" #
    gfn_samples_path = f'{hps.gfn_samples_path}/GFN_gen_samples_{hps.target_name}/'

    os.makedirs(gfn_samples_path, exist_ok=True)

    finetuner = DockingFineTuner(hps,conditional_range_dict, cond_prop_var, hps['saved_model_path'],rank=rank, world_size= world_size, gfn_samples_path=gfn_samples_path)
    train_loader = build_train_loader(hps, finetuner)
    wandb_config = {"Note": f"Finetuning GFN, pretrained GFN {hps['saved_model_path']},  task {hps['task']}; same condition and reward (+task rew) as pretrained GFN, layerwise LR, molwt 160_300, conditional changed, molwt_slope=0; fname: gfn_finetune_molwt_clearshadow_mwt_160_300_updatedcond_molwt_slope_0.out"}

    wandb_config.update(hps)
    train(hps, finetuner, train_loader,rank ,wandb_config, wandb_project_name,  run_name=f'{hps.target_name}_RTB_Denovo',wandb_mode=None,world_size=world_size) #"disabled"

if __name__ == "__main__":
    world_size = 1 # tested only with world_size 1 
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    main(0,1)