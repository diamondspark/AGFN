import argparse
import pandas as pd
from utils.metrics_utils import get_metrics, TaskPredictor
from reward import RewardFineTune
import torch
from distutils.util import strtobool
from agfn.sampling import Sampler
import numpy as np

def metrics_pre_prep(args):
    loaded_dict = torch.load(args.model_path,map_location='cpu')
    hps = loaded_dict['hps']
    cond_prop_var= {
            'tpsa':20,
            'num_rings':1,
            'sas':1,
            'qed':1,
            }
    conditional_range_dict = dict()
    for k in cond_prop_var:
        conditional_range_dict[k] = hps[k]

    reward_ft =  RewardFineTune(conditional_range_dict,None, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'], hps)

    sampler = Sampler(args.model_path)
    sampled_mols, sampled_smiles = sampler.sample_smiles(args.ntrajs)
    
    online_rew_tup = reward_ft.molecular_rewards(sampled_mols)
    online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
    normalized_task_rew_online, true_task_score_online = reward_ft.task_reward(hps.task, reward_ft.task_model, sampled_mols) #for TDC

    if hps.task_rewards_only:
        flat_rewards_online =  torch.Tensor(normalized_task_rew_online)
    else:
        flat_rewards_online = online_rew*normalized_task_rew_online

    
    res = {}
    res['normalized_moo_rew']= online_rew_tup[0] if  hps.objective=='property_targeting' else np.hstack([online_rew_tup[0],normalized_task_rew_online]) 
    res['overall_rew'] = flat_rewards_online
    res['generated_smiles_list'] = sampled_smiles
    res['pretrain_conditionals']= conditional_range_dict
    res['task_conditionals'] = {hps['task']: [hps.task_possible_range, hps.task_possible_range, hps.pref_dir]} #task_conditional_range_dict if task_conditional_range_dict else {}
    res['hps']= hps
    
    return res
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the model checkpoint")
    parser.add_argument('--ntrajs', type=int, default=100, help="Number of trajectories to sample for metric computation")
    parser.add_argument('--pretrain_smiles_path', default='./data/250k_rndm_zinc_drugs_clean_3.csv', type=str, help="Path to the pretrain SMILES CSV file")
    parser.add_argument('--do_hyp_vol', default='False', type=str, help="Flag to determine whether to do hypersphere volume calculation")

    args = parser.parse_args()
    res = metrics_pre_prep(args)
    
    task_model_paths = {
    'LD50': './saved_models/task_models/LD50_Zhu_mae0.623_catboostmaplight.pkl',
    'Caco2': './saved_models/task_models/caco2_wang_mae0.273_catboostmaplight.pkl',
    'Lipophilicity': './saved_models/task_models/Lipophilicity_AstraZeneca_mae0.545_catboostmaplight.pkl',
    'Solubility': './saved_models/task_models/Solubility_AqSolDB_mae0.79_catboostmaplight.pkl',
    'BindingRate': './saved_models/task_models/PPBR_AZ_mae7.64_catboostmaplight.pkl',
    'MicroClearance': './saved_models/task_models/Clearance_Microsome_AZ_spearman0.564_catboostmaplight.pkl',
    'HepatocyteClearance': './saved_models/task_models/Clearance_Hepatocyte_AZ_spearman0.473_catboostmaplight.pkl'
    }   
    
    do_hyp_vol = bool(strtobool(args.do_hyp_vol))

    pretrain_smiles_list = pd.read_csv(args.pretrain_smiles_path).smiles.to_list()
    gen_smi_overall_rew = res['overall_rew']
    gen_smi_moo_rew = torch.Tensor(res['normalized_moo_rew'])
    generated_smiles_list = res['generated_smiles_list']
    pretrain_conditionals = res['pretrain_conditionals'] #{} for prop_opt single objective optimization; all_conditionals in metrics is task conditionals
    task_conditionals = res['task_conditionals']
    metrics = get_metrics(
        generated_smiles_list, gen_smi_overall_rew,
        pretrain_conditionals, pretrain_smiles_list,
        task_conditionals= task_conditionals, predictor=TaskPredictor(task_model_paths),
          gen_smi_moo_rew= gen_smi_moo_rew, hypervolume= do_hyp_vol, task_offline_df=None, binding=False, protein=None, docking_scores=None
    )
    print(metrics)