# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/gfn_pretrain_test_env_public/src')
from pretrainSrc.pretrainer import Pretrainer
import os
import time
from trajmixing import TrajectoryMixer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from gflownet.algo.tb_loss import compute_batch_losses
import wandb
import numpy as np
from rdkit.Chem import QED
from rdkit import Chem
from easydict import EasyDict
import yaml
from iterators.samp_iter_inmem import build_train_loader
# from samp_iter.samp_iter_inmem import build_train_loader

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
wandb.login()

def do_validation(reward, online_mols, offline_smiles):
    qed_scores_on = np.array([QED.qed(mol) for mol in online_mols])
    offline_mols = [Chem.MolFromSmiles(smiles) for smiles in offline_smiles]
    qed_scores_off = np.array([QED.qed(mol) for mol in offline_mols])
    avg_qed_score_on, avg_qed_score_off = np.mean(qed_scores_on), np.mean(qed_scores_off)
    return avg_qed_score_on, avg_qed_score_off



def train(hps, pretrainer, mixer, train_loader, rank, wandb_config, wandb_project_name, wandb_mode="disabled",
          world_size=1):
    with wandb.init(project = wandb_project_name, config = wandb_config, mode = wandb_mode,
                    group='ddp'):
        print(wandb.run.id, wandb.run.name)
        if not os.path.exists(hps["log_dir"]+f'/{wandb.run.name}/'):
            os.makedirs(hps["log_dir"]+f'/{wandb.run.name}/')
            tmppath = hps["log_dir"]+f'/{wandb.run.name}/'
            print(f'successfully made logging dir {tmppath}')
        
        hps["log_dir"] = hps["log_dir"]+f'/{wandb.run.name}/'

        cumtime = 0
        tstart = t0 = None # time.time()
        for i, (gfn_batch, avg_traj_len, zinc_filelog, online_mols, offline_smiles) in enumerate(train_loader):
            try:
                if tstart is None: # skip the first batch + process init
                    tstart = t0 = time.time()
                t1 = time.time()
                wait_time = t1 - t0
                it = pretrainer.start_step+i
                loss, info = compute_batch_losses(pretrainer.algo, pretrainer.gfn_trainer,
                                                gfn_batch.to(pretrainer.device),
                                                pretrainer.device, hps)
                step_info = pretrainer.gfn_trainer.step(loss)
                loss_val = loss.item()

                info_vals = {"Total loss ":loss_val,
                            "logZ":info['logZ'],
                            "online_loss": info['online_loss'],
                            "offline_loss": info['offline_loss'],
                            "total_gfn_loss": info['total_gfn_loss'],
                            "mle_loss":info['mle_loss'],
                            "avg_batch_fwd_traj_len":avg_traj_len[0],
                            "avg_batch_bck_traj_len":avg_traj_len[1],
                            "avg_batch_traj_len":avg_traj_len[2],
                            "Average Reward": torch.mean(gfn_batch.flat_rewards).item(),
                            "Average online reward": torch.mean(gfn_batch.online_flat_rewards).item(),
                            "Average offline reward": torch.mean(gfn_batch.offline_flat_rewards).item(),
                            "Percent_valid_mols":gfn_batch.valid_percent,
                            "Zinc Radius Offline": gfn_batch.offln_zinc_rad,
                            "Zinc Radius Online": gfn_batch.onln_zinc_rad,
                            "Avg. Num_Rings Online": gfn_batch.online_num_rings,
                            "Avg. Num_Rings Offline": gfn_batch.offln_num_rings,
                            "Avg. tpsa Online": gfn_batch.online_avg_tpsa,
                            "Avg. tpsa Offline": gfn_batch.offline_avg_tpsa,
                            "Avg. QED onln":gfn_batch.online_qed,
                            "Avg. QED offln": gfn_batch.offline_qed,
                            "Avg. SAS onln":gfn_batch.online_avg_SAS,
                            "Avg. SAS offln": gfn_batch.offline_avg_SAS,
                            "Train_iter":i}
                
                if (it %1000 == 0):
                    # print("Syncing with other processes...", it)
                    avg_qed_score_on, avg_qed_score_off= do_validation(pretrainer.reward, online_mols, offline_smiles)
                    validation_info_vals = {"Val onln Avg QED": avg_qed_score_on,
                        "Val offln Avg QED": avg_qed_score_off}
                    info_vals.update(validation_info_vals)
                    
                keys = sorted(info_vals.keys())
                if i%10==0:
                    print(f' iter {i} ---> Loss {loss_val}')
                    # if rank ==0:
                    #     with open(hps["log_dir"] + f"/gen_mols_{it}.txt", "w") as f:
                    #         content = f"Zinc Radius Online {np.average(gfn_batch.onln_zinc_rad)} Val onln Avg QED: {avg_qed_score_on} \n {str([Chem.MolToSmiles(mol) for mol in online_mols])}"
                    #         f.write(content)
                if world_size > 1:
                    all_info_vals = torch.zeros(len(keys)).to(rank)
                    for i, k in enumerate(keys):
                        all_info_vals[i] = info_vals[k]
                    dist.all_reduce(all_info_vals, op=dist.ReduceOp.SUM)
                    for i, k in enumerate(keys):
                        info_vals[k] = all_info_vals[i].item() / world_size
                    if rank == 0:
                        wandb.log(info_vals)
                else:
                    wandb.log(info_vals)
                t0 = time.time()
                step_time = t0 - t1
                cumtime += step_time
                tottime = t0 - tstart
                
                if (it % 100== 0) and (world_size > 1):
                    # print("Syncing with other processes...")
                    dist.barrier()
                # if it == 200:
                #     break
                # if (it > 0 )and (it %  hps['checkpoint_every'] == 0) and (pretrainer.rank==0):
                if (it %  hps['checkpoint_every'] == 0) and (pretrainer.rank==0):
                    #TODO: update _save_state() to save optimizer's state as well
                    pretrainer.gfn_trainer._save_state(it,wandb.run.name)
            except Exception as e:
                print (e)
                raise(e)
                continue
    if world_size > 1:
        dist.destroy_process_group()

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12376'
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    with open('./src/config/pretrain.yml', 'r') as f:
        hps = EasyDict(yaml.safe_load(f))
    hps['offline_data']= True
    
    #default_hps needed for GFN_trainer. Merging together with hps
    default_hps= {"Z_learning_rate": 1e-3,}
   
    hps.update(default_hps)
    # Suggested ranges from 
    # https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/64e5333e3fdae147faa01202/original/stoplight-a-hit-scoring-calculator.pdf
    # First list is drug like range, second list is lower and upper bounds for thermometer encoding, +/-1 indicates if higher/lower is desirable.
    # Latter are more liberal bounds
    conditional_range_dict = {
                            'tpsa':[[60,100],[10,200], 0],            #[120,140],  #lower is better, open lower bound
                            'num_rings':[[1,3],[1,5],1], # Num of 5 or 6 membered rings only
                            'sas': [[1,3],[1,5], 0], 
                            #'zinc_radius':[[0.5,1],[0,1], 1],            # range needed for code stability. Not conditioing zinc reward on these bounds
                            'qed':[[0.65,0.8],[0,1], 0],
                            }
    cond_prop_var= {
                    'tpsa':20,
                    'num_rings':1,
                    'sas':1,
                    'qed':1,
                    }
    hps.update(conditional_range_dict)


    wandb_project_name = "GFN_Pretrain"
    if hps['load_saved_model']:
        pretrainer = Pretrainer(hps, conditional_range_dict, cond_prop_var, rank=rank, world_size=world_size, load_path=hps['model_load_path'])
    else:
        pretrainer = Pretrainer(hps, conditional_range_dict,cond_prop_var, rank=rank, world_size=world_size)
    trainable_params = sum(p.numel() for p in pretrainer.gfn_trainer.model.parameters() if p.requires_grad)
    num_online, num_offline = pretrainer.mixing_counts(hps['sampling_batch_size'],hps['mix_ratio'])
    mixer = TrajectoryMixer(pretrainer,num_online, num_offline)

    # from Samp_iter_offline import build_train_loader
    # from Samp_iter_online import build_train_loader
    train_loader = build_train_loader(hps, pretrainer, mixer, datafile = hps['offline_data_file']) # 'zinc_1M_random_df.csv')#'250k_rndm_zinc_drugs_clean_3.csv')

    wandb_config = {"Note": f" PreTraining GFN w/ inexpensive reward"}
    wandb_config.update(pretrainer.hps)
    train(hps, pretrainer, mixer, train_loader, rank, wandb_config, wandb_project_name, wandb_mode= None, world_size=world_size)
    return

if __name__ == "__main__":
    world_size = 1 #torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        print('driver.py spawned')
    else:
        main(0,1)


