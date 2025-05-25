import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi
import torch
import os
import argparse
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models.conditionals import ConditionalInfo
from gflownet.models.graph_transformer import GraphTransformerGFN, PrunedGraphTransformerGFN
from gflownet.algo.graph_sampling import GraphSampler
from tqdm import tqdm

# from MyGFN_Pretrain.Experiments.BindingMols.Denovo_results import (
#     DockingFineTuneSampler, get_metrics, get_overall_reward, 
#     dock_with_min_score, pretrain_smiles_list
# )


def get_inchi_key(mol, stereo):
    """Get InChI key of a molecule (a unique identifier of molecule). Optionally ignore stereochemistry and isotopes."""
    inchi_key = inchi.MolToInchiKey(mol)
    if not stereo:
        q = inchi_key.split('-')
        inchi_key = q[0] + '-' + q[2]  # Remove middle part responsible for stereo and isotopes
    return inchi_key

def remove_duplicates(smiles_list, reference_smiles=None, stereo=False):
    """
    Remove duplicate SMILES based on their InChI keys.

    Parameters:
    smiles_list: list of SMILES strings to process.
    reference_smiles: list of reference SMILES (duplicates compared to these will be removed).
    stereo: If True, consider stereochemistry and isotopes during duplicate detection.

    Returns:
    A tuple of (unique_smiles, duplicates) where:
    - unique_smiles: list of unique SMILES
    - duplicates: list of tuples (original_smiles, duplicate_smiles)
    """
    d = dict()  # key: InChI key, value: list of SMILES
    unique_smiles = []
    duplicates = []

    # Convert reference SMILES to InChI keys if provided
    if reference_smiles is not None:
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchi_key = get_inchi_key(mol, stereo)
                d[inchi_key] = [smiles]

    # Process input SMILES
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchi_key = get_inchi_key(mol, stereo)
            if inchi_key not in d:
                d[inchi_key] = [smiles]
                unique_smiles.append(smiles)
            else:
                d[inchi_key].append(smiles)
                duplicates.append((d[inchi_key][0], smiles))  # Add first encountered SMILES and current duplicate

    return unique_smiles, duplicates


class Sampler():
    def __init__(self, model_load_path, seed_graph= None, save_path = './data/gfn_samples/'):
        self.device = torch.device(f'cuda') if torch.cuda.is_available() is not None else torch.device('cpu')
        self.seed_graph = seed_graph
        self.save_path = save_path
        loaded_dict = torch.load(model_load_path, map_location='cpu')
        self.hps = loaded_dict['hps']
        self.cond_prop_var= {
            'tpsa':20,
            'num_rings':1,
            'sas':1,
            'qed':1,
            }
        self.conditional_range_dict = dict()
        for k in self.cond_prop_var:
            self.conditional_range_dict[k] = self.hps[k]
        rng = np.random.default_rng(self.hps['random_seed'])
        self.ctx =  MolBuildingEnvContext(num_cond_dim=2*self.hps["num_thermometer_dim"]*(len(self.conditional_range_dict)-1), #chiral_types=[ChiralType.CHI_UNSPECIFIED],
                                          charges=self.hps.get('charges',[0]), 
                                          atoms = self.hps.get('atoms',["C", "N", "O", "F", "P", "S"]), 
                                          num_rw_feat=0) 
        self.ctx.num_rw_feat = 0
        env = GraphBuildingEnv()
        if hasattr(self.ctx, "graph_def"):
            env.graph_cls = self.ctx.graph_cls
        algo = TrajectoryBalance(env,self.ctx,rng,self.hps) 
        # self.gfn_trainer = GFNTrainerRTB(self.hps, algo, rng, self.device, env, self.ctx) if self.hps['type'].lower()=='rtb' else GFNTrainer(self.hps, algo, rng, self.device, env, self.ctx)
        self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"],
                                         do_bck=self.hps['tb_p_b_is_parameterized'], num_graph_out=int(self.hps["tb_do_subtb"]), num_mlp_layers=self.hps['num_mlp_layers'])
        if self.hps['type'].lower()=='rtb':
            self.model = PrunedGraphTransformerGFN(self.model, self.ctx, self.hps)
        self.graph_sampler = GraphSampler(self.ctx, env, self.hps["max_traj_len"],self.hps["max_nodes"],rng,self.hps["sample_temp"],
                                correct_idempotent=False, pad_with_terminal_state=True)
        
            
        model_dict = loaded_dict['models_state_dict'][0]
        tmp_flag = False
        for key in model_dict:
            if 'module' in key:
                new_model_dict= {}
                for k in model_dict:
                    val = model_dict[k]
                    k_new = k.replace('module.','')
                    new_model_dict[k_new] = val
                self.model.load_state_dict(new_model_dict)
                # self.gfn_trainer.model_prior.load_state_dict(new_model_dict)
                tmp_flag = True
                break
        if not tmp_flag:
            self.model.load_state_dict(loaded_dict['models_state_dict'][0]) 
            # self.gfn_trainer.model_prior.load_state_dict(loaded_dict['models_state_dict'][0]) 
        self.model.to(self.device)
        print("len loaded dict:", len(loaded_dict['models_state_dict'][0]))
        self.cond_info_cmp = ConditionalInfo(self.conditional_range_dict, self.cond_prop_var, self.hps['num_thermometer_dim'], self.hps['OOB_percent'])
        
        
    # def sample_smiles(self, n_trajs, bs = 32, dedup=True):
    #     cond_info = self.cond_info_cmp.compute_cond_info_forward(n_trajs)
    #     cond_info_encoding = self.cond_info_cmp.thermometer_encoding(cond_info).to(self.device)
        
    #     cur_result =  self.graph_sampler.sample_from_model(self.gfn_trainer.model,n_trajs,
    #                                                 cond_info_encoding,
    #                                                 self.device,
    #                                                 random_stop_action_prob= self.hps['random_stop_prob'],
    #                                                 random_action_prob = self.hps['random_action_prob'],
    #                                                 seed_graph= self.seed_graph
    #                                                 )
    #     valid_idcs = torch.tensor([i + 0 for i in range(len(cur_result)) if (cur_result[i + 0]["is_valid"]) & ("fwd_logprob" in cur_result[i+0])]).long()
    #     mols = [self.ctx.graph_to_mol(cur_result[i]['traj'][-1][0]) for i in valid_idcs]
    #     smiles= [Chem.MolToSmiles(mol) for mol in mols]
    #     return mols, smiles
    
    def sample_smiles(self, n_trajs, bs=32, dedup=True, save_dir="./data/gfn_samples/smiles_checkpoints"):
        os.makedirs(save_dir, exist_ok=True)

        all_smiles = []
        all_mols = []
        seen_smiles = set()

        pbar = tqdm(total=n_trajs, desc="Sampling unique SMILES", dynamic_ncols=True)

        # Checkpointing config only if n_trajs > 1000
        enable_checkpointing = n_trajs > 1000
        percent_interval = 10
        next_save_threshold = percent_interval
        last_saved_file = None

        while len(all_smiles) < n_trajs:
            current_bs = min(bs, n_trajs - len(all_smiles))
            cond_info = self.cond_info_cmp.compute_cond_info_forward(current_bs)
            cond_info_encoding = self.cond_info_cmp.thermometer_encoding(cond_info).to(self.device)

            cur_result = self.graph_sampler.sample_from_model(
                self.model,
                current_bs,
                cond_info_encoding,
                self.device,
                random_stop_action_prob=self.hps['random_stop_prob'],
                random_action_prob=self.hps['random_action_prob'],
                seed_graph=self.seed_graph
            )

            valid_idcs = [
                i for i in range(len(cur_result))
                if cur_result[i]["is_valid"] and "fwd_logprob" in cur_result[i]
            ]

            mols = [self.ctx.graph_to_mol(cur_result[i]['traj'][-1][0]) for i in valid_idcs]
            smiles = [Chem.MolToSmiles(mol) for mol in mols]

            if dedup:
                unique_smiles, _ = remove_duplicates(smiles, reference_smiles=all_smiles, stereo=False)
                mols = [mol for smi, mol in zip(smiles, mols) if smi in unique_smiles]
                smiles = unique_smiles

            added = 0
            for smi, mol in zip(smiles, mols):
                if len(all_smiles) >= n_trajs:
                    break
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    all_smiles.append(smi)
                    all_mols.append(mol)
                    added += 1

            pbar.update(added)

            # Checkpoint saving if enabled
            if enable_checkpointing:
                current_percent = int(100 * len(all_smiles) / n_trajs)
                if current_percent >= next_save_threshold:
                    checkpoint_path = os.path.join(save_dir, f"smiles_{next_save_threshold}pct.pkl")
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(all_smiles, f)

                    if last_saved_file and os.path.exists(last_saved_file):
                        os.remove(last_saved_file)

                    last_saved_file = checkpoint_path
                    next_save_threshold += percent_interval

        pbar.close()

        # Save final full SMILES list
        final_path = os.path.join(save_dir, "smiles_final.pkl")
        with open(final_path, "wb") as f:
            pickle.dump(all_smiles, f)
        print(f'Saved {len(all_smiles)} samples to ./data/gfn_samples/smiles_checkpoints/smiles_final.pkl')
        return all_mols, all_smiles

# class FineTuneSampler(FineTuner):
#     def __init__(self, hps, conditional_range_dict, cond_prop_var, load_path, rank, world_size, ft_conditionals_dict=None):
#         super().__init__(hps, conditional_range_dict, cond_prop_var, load_path, rank, world_size, ft_conditionals_dict)
#         loaded_dict = torch.load(load_path,map_location='cpu')
#         model_dict = loaded_dict['models_state_dict'][0]
#         new_model_dict= {}
#         for k in model_dict:
#             val = model_dict[k]
#             k_new = k.replace('module.','')
#             new_model_dict[k_new] = val

#         self.gfn_trainer.model.load_state_dict(new_model_dict)
#         self.gfn_trainer.model.to(self.device)
#         self.task = self.cond_info_task = ConditionalInfo(conditional_range_dict, cond_prop_var, hps['num_thermometer_dim'], hps['OOB_percent'])

#     def sample_smiles(self, traj_sampler, n_trajs):
#         _,cur_result = traj_sampler.sample_tau_from_PF(self.gfn_trainer.model,self.device,n_trajs=n_trajs)
#         valid_idcs = torch.tensor([i + 0 for i in range(len(cur_result)) if (cur_result[i + 0]["is_valid"]) & ("fwd_logprob" in cur_result[i+0])]).long()
#         mols = [self.ctx.graph_to_mol(cur_result[i]['traj'][-1][0]) for i in valid_idcs]
#         smiles= [Chem.MolToSmiles(mol) for mol in mols]
#         return mols, smiles



# def run_sampling(target='jak2', n_trajs=100, hit_thr=9.1, save_path=None):
#     hps = build_hyperparams(target)
#     ft_sampler = DockingFineTuneSampler(hps, CONDITIONAL_RANGE_DICT, COND_PROP_VAR, hps.model_load_path, rank=0, world_size=1)
#     traj_sampler = Trajectory_Sampling(ft_sampler)

#     sampled_mols, sampled_smiles = ft_sampler.sample_smiles(traj_sampler, n_trajs)
#     minimized_doc_res = dock_with_min_score(VINA_PATH, target, sampled_smiles, num_trials=1)

#     below_thresh_count = np.sum(np.array(minimized_doc_res[1]) < hit_thr)
#     overall_rew = get_overall_reward(minimized_doc_res[0], minimized_doc_res[2])

#     metrics = get_metrics(
#         sampled_smiles,
#         overall_rew,
#         CONDITIONAL_RANGE_DICT,
#         pretrain_smiles_list,
#         {},
#         hypervolume=False,
#         binding=True,
#         protein=target,
#         docking_scores=minimized_doc_res[1]
#     )
#     metrics.update({'below_thresh_count': int(below_thresh_count)})
#     print(f"[{target.upper()}] Sampling Results:", metrics)

#     if save_path:
#         with open(save_path, 'wb') as f:
#             pickle.dump([sampled_smiles, minimized_doc_res[1]], f)

#     return sampled_smiles, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample molecules using a trained GFlowNet sampler.")
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("n_trajs", type=int, help="Number of SMILES to sample.")
    parser.add_argument("--bs", type=int, default=32, help="Batch size for sampling.")
    parser.add_argument("--save_dir", type=str, default="./data/gfn_samples/smiles_checkpoints", help="Directory to save sampled SMILES.")

    args = parser.parse_args()

    sampler = Sampler(model_load_path=args.model_path)
    mols, smiles = sampler.sample_smiles(
        n_trajs=args.n_trajs,
        bs=args.bs,
        save_dir=args.save_dir
    )
