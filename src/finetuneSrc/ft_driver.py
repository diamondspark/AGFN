import yaml
from easydict import EasyDict
import argparse
import torch
from utils.maplight import *
from gflownet.algo import rtb_loss as rtb 
from gflownet.algo import tb_loss as tb 
from iterators.samp_iter_finetune import build_train_loader
from finetuneSrc.finetuner import FineTunerRTB, FineTuner
from torch import nn as nn
import os
from rdkit import RDLogger, Chem
import torch.multiprocessing as mp
import torch.distributed as dist
import time
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
RDLogger.DisableLog("rdApp.*")
import wandb
wandb.login()

def train(hps, trainer, train_loader, rank, wandb_config, wandb_project_name, run_name, wandb_mode=None, world_size=1):
    t0 = time.time()
    with wandb.init(project = wandb_project_name, config = wandb_config, mode = wandb_mode):#) config = wandb_config, mode = wandb_mode, id = "yrt9nv2g",resume="must"):
        wandb.run.name=f'{run_name}_{wandb.run.name}'
        print('Wandb Run: ',wandb.run.id, wandb.run.name)
        if not os.path.exists(hps["log_dir"]+f'/{wandb.run.name}/'):
            os.makedirs(hps["log_dir"]+f'/{wandb.run.name}/')
        hps["log_dir"] = hps["log_dir"]+f'/{wandb.run.name}/'
        for i, (gfn_batch, mols) in enumerate(train_loader):
            try:
                it = trainer.start_step+i
                if hps.type == 'rtb':
                    loss, info = rtb.compute_batch_losses(trainer.algo, trainer.gfn_trainer,
                                                gfn_batch.to(trainer.device), trainer.device, i)
                elif hps.type == 'finetuning':
                    loss, info = tb.compute_batch_losses(trainer.algo, trainer.gfn_trainer,
                                                gfn_batch.to(trainer.device), trainer.device, hps)
                step_info = trainer.gfn_trainer.step(loss)
                loss_val = loss.item()
                # print(f' iter {i} ---> Loss {loss_val} mean_Rewards {torch.mean(gfn_batch.flat_rewards)}, avg_traj_len {gfn_batch.avg_batch_len}, logZ {info["logZ"]}')
                if hps.offline_data:
                    info_vals = {"Total loss ":loss_val,
                        "logZ":info['logZ'],
                        "online_loss": info['online_loss'],
                        "offline_loss": info['offline_loss'],
                        "total_gfn_loss": info['total_gfn_loss'],
                        "mle_loss":info['mle_loss'],
                        "Average Overall Reward": torch.mean(gfn_batch.flat_rewards).item(),
                        "Average Overall Online Reward": torch.mean(gfn_batch.online_flat_rewards).item(),
                        "Average Overall Offline Reward": torch.mean(gfn_batch.offline_flat_rewards).item(),
                        "Percent_valid_mols":gfn_batch.valid_percent,
                        "Percent_unique_in_batch":gfn_batch.unique_percent,
                        "Average traj len online": gfn_batch.avg_batch_len,
                        "Zinc Radius Offline": gfn_batch.offln_zinc_rad,
                        "Zinc Radius Online":gfn_batch.onln_zinc_rad,
                        "Avg. Num_Rings Online": gfn_batch.online_num_rings,
                        "Avg. Num_Rings Offline": gfn_batch.offln_num_rings,
                        "Avg. tpsa Online": gfn_batch.online_avg_tpsa,
                        "Avg. tpsa Offline": gfn_batch.offline_avg_tpsa,
                        "Avg. QED onln":gfn_batch.online_qed,
                        "Avg. QED Offline": gfn_batch.offline_qed,
                        "Avg. SAS onln":gfn_batch.online_avg_SAS,
                        "Avg. SAS Offline": gfn_batch.offline_avg_SAS,
                        f"Average norm {hps['task']} rew online":gfn_batch.avg_task_reward_online,
                        f"Average {hps['task']} score online": gfn_batch.avg_task_score_online,
                        f"Average norm {hps['task']} rew offline":gfn_batch.avg_task_reward_offline,
                        f"Average {hps['task']} score offline": gfn_batch.avg_task_score_offline,
                        "Average fwd_logprob": gfn_batch.avg_fwd_logprob.item(),
                        "Average bck_logprob": gfn_batch.avg_bck_logprob.item(),
                        "Train_iter":i,
                        "Time elapsed":time.time()-t0
                        }
                else:
                    info_vals = {"Total loss ":loss_val,
                        "logZ":info['logZ'],
                        "online_loss": info['online_loss'],
                        "Average Overall Reward": torch.mean(gfn_batch.flat_rewards).item(),
                        f"Average norm {hps['task']} rew online":gfn_batch.avg_task_reward,
                        f"Average {hps['task']} score online": gfn_batch.avg_task_score,
                        "Percent_valid_mols":gfn_batch.valid_percent,
                        "Percent_unique_in_batch":gfn_batch.unique_percent,
                        "Average traj len online": gfn_batch.avg_batch_len,
                        "Average fwd_logprob": gfn_batch.avg_fwd_logprob.item(),
                        "Average bck_logprob": gfn_batch.avg_bck_logprob.item(),
                        "Avg. tpsa Online": gfn_batch.avg_tpsa,
                        "Avg. Num_Rings Online":gfn_batch.avg_num_rings,
                        "Avg. SAS onln": gfn_batch.avg_sas,
                        "Avg. QED onln": gfn_batch.avg_qed,
                        "Zinc Radius Online":gfn_batch.avg_zinc_rad,
                        "Train_iter":i,
                        "Time elapsed":time.time()-t0}
                
                if (hps.get('objective', None)=='property_targeting'):
                    if(hps.subtype=='new_props'):
                        info_vals.update({f"Avg new prop {hps.new_props.added_prop}": gfn_batch.avg_new_prop }) 
                                
                if gfn_batch.valid_percent ==0:
                    print('info vals ',info_vals)

                if it%10 ==0:
                    print(f' iter {i} ---> Loss {loss_val} mean_Rewards {torch.mean(gfn_batch.flat_rewards)}, avg_traj_len {gfn_batch.avg_batch_len}, logZ {info["logZ"]}')

                keys = sorted(info_vals.keys())

                if i%100==0:
                    if rank ==0:
                        with open(hps["log_dir"] + f"/gen_mols_{it}.txt", "w") as f:
                            if hps.offline_data:
                                content = f"Average QED {gfn_batch.online_qed} Val Zinc Score: {gfn_batch.onln_zinc_rad} \n {str([Chem.MolToSmiles(mol) for mol in mols])}"
                            else:
                                content = f"Average QED {gfn_batch.avg_qed} Val Zinc Score: {gfn_batch.avg_zinc_rad} \n {str([Chem.MolToSmiles(mol) for mol in mols])}"
                            f.write(content)
                            f.close()

                if world_size > 1:
                    all_info_vals = torch.zeros(len(keys)).to(rank)
                    for i, k in enumerate(keys):
                        try:
                            all_info_vals[i] = info_vals[k]
                        except Exception as e:
                            print(e,k)
                            raise e
                    dist.all_reduce(all_info_vals, op=dist.ReduceOp.SUM)
                    for i, k in enumerate(keys):
                        info_vals[k] = all_info_vals[i].item() / world_size
                    if rank == 0:
                        wandb.log(info_vals)
                else:
                    wandb.log(info_vals)

                
                if (it % 100== 0) and (world_size > 1):
                        dist.barrier()

                if (it %  hps['checkpoint_every']==0) and (trainer.rank == 0): #trainer.gfn_trainer.ckpt_scheduler.should_save() and  (trainer.rank == 0)  :#  
                    trainer.gfn_trainer._save_state(it-trainer.start_step,wandb.run.name)
            except Exception as e:
                print(e)
                raise(e)
                continue
    if world_size > 1:
        dist.destroy_process_group()
    return


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12388'
    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # # Define task ranges for each variant
    # task_ranges = {
    #     "LD50": {
    #         "preserved": (2.3435, 5.541),
    #         "loTPSA": (2.448, 5.513),
    #         "hiTPSA": (2.12, 4.38),
    #     },
    #     "Caco2": {
    #         "preserved": (-4.7544875, -4.059999899999998),
    #         "loTPSA": (-4.5968795, -4.0100002),
    #         "hiTPSA": (-5.6399999, -4.9499998),
    #     },
    #     "HepatocyteClearance": {
    #         "preserved": (3.0, 13.18),
    #         "loTPSA": (3.0, 38.33),
    #         "hiTPSA": (3.0, 20.0),
    #     },
    #     "MicroClearance": {
    #         "preserved": (5.0, 15.0),
    #         "loTPSA": (5.0, 15.0),
    #         "hiTPSA": (5.0, 15.0),
    #     },
    #     "BindingRate": {
    #         "preserved": (10.09, 95.77),
    #         "loTPSA": (11.18, 93.94),
    #         "hiTPSA": (62.4, 95.72),
    #     },
    #     "Solubility": {
    #         "preserved": (-2.798, 0.623),
    #         "loTPSA": (-3.085, 0.8177),
    #         "hiTPSA": (-2.71, 0.5402),
    #     },
    #     "Lipophilicity": {
    #         "preserved": (-1.45, 4.48),
    #         "loTPSA": (-1.31, 4.29),
    #         "hiTPSA": (-1.2, 3.7),
    #     },
    # }



    parser = argparse.ArgumentParser(description="Run experiment configurations.")
    parser.add_argument('config', type=str, help = "Path for relevant config.yml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    hps = config.finetuning
    default_hps= {"Z_learning_rate": 1e-3,}
    hps.update(default_hps)

    # Print experiment configuration
    print(f"Task: {hps['task']}")
    print(f"Task Range: {hps['task_possible_range']}")
    print(f"Task Slope: {hps['pref_dir']}")

    if hps.offline_data:
        hps.offline_df_path= hps.offline_df_path 


    conditional_range_dict = {
                            'tpsa':[[60,100],[10,200], 0],            #[120,140],  #lower is better, open lower bound
                            'num_rings':[[1,3],[1,5],1], 
                            'sas': [[1,3],[1,5], 0], 
                            'qed':  [[0.65,0.8],[0,1], 0]                                    # [[0.65,0.8],[0,1], 0], Zn1M
                            }

    cond_prop_var= {
                'tpsa':20,
                'num_rings':1,
                'sas':1,
                'qed':1,
                }

    if hps.objective=='property_optimization':
        conditional_range_dict = {
                            'tpsa':[[10,200],[10,200], 0], 
                            'num_rings':[[1,5],[1,5],1], 
                            'sas': [[1,5],[1,5], 0], 
                            'qed':  [[0,1],[0,1], 0]                                    # [[0.65,0.8],[0,1], 0], Zn1M
                            }
        # hps.task_conditionals = False
        # task_conditional_range_dict = None
        hps.task_rewards_only = True

    elif hps.objective=='property_targeting':
        # if hps.subtype=='DRA':
        conditional_range_dict.update(hps.value_mod.updated_prop)
            # hps.task_conditionals = False
            # task_conditional_range_dict = None
    
    elif hps.objective=='property_constrained_optimization':
        if hps['subtype']=='preserved':
            # hps.task_conditionals= False
            # task_conditional_range_dict = None
            hps.task_rewards_only = False
            pass
        elif hps['subtype']=='DRA':
            conditional_range_dict.update(hps.value_mod.updated_prop)
            hps.task_rewards_only = False
        # elif hps['subtype']=='loTPSA':
        #     conditional_range_dict.update({'tpsa':[[40,60],[10,200], 0]})
        #     hps.task_rewards_only = False
        #     pass
        # elif hps['subtype']=='hiTPSA':
        #     conditional_range_dict.update({'tpsa':[[100,120],[10,200], 0]})
        #     hps.task_rewards_only = False
        #     pass

    hps.update(conditional_range_dict)

    hps.task_model_path = hps.task_model_path 

    wandb_project_name = hps.get("wandb_project_name", "GFN_Finetune") #"GFN_leadoptim" #
    if hps.type == 'rtb':
        finetuner = FineTunerRTB(hps,conditional_range_dict, cond_prop_var, hps['saved_model_path'],rank, world_size)
    elif hps.type == 'finetuning':
        finetuner = FineTuner(hps,conditional_range_dict, cond_prop_var, hps['saved_model_path'],rank, world_size)
    train_loader = build_train_loader(hps, finetuner)
    wandb_config = {"Note": f"Finetuning GFN, pretrained GFN {hps['saved_model_path']},  task {hps['task']}; same condition and reward (+task rew) as pretrained GFN, layerwise LR, molwt 160_300, conditional changed, molwt_slope=0; fname: gfn_finetune_molwt_clearshadow_mwt_160_300_updatedcond_molwt_slope_0.out"}

    wandb_config.update(hps)

    print('conditional range dict ', conditional_range_dict)
    run_type = "RTB_" if hps.type.lower() == "rtb" else ""
    run_name = f"{run_type}FT_{hps.task}_{hps.objective}"
    train(hps, finetuner, train_loader,rank ,wandb_config, wandb_project_name,  run_name=run_name,wandb_mode=None,world_size=world_size) #"disabled"

if __name__ == "__main__":
    world_size = 1 #torch.cuda.device_count() 
    print('Let\'s use', world_size, 'GPUs!')
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    main(0,1)