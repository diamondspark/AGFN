import torch
from gflownet.envs.graph_building_env import GraphActionCategorical
import datamol as dm
import os
from fnmatch import fnmatch
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def validate_batch( batch, trajs, ctx, model):
    #borrowed from current trunk, sampling_it.py 
    for actions, atypes in [(batch.actions, ctx.action_type_order)] + (
        [(batch.bck_actions, ctx.bck_action_type_order)]
        if hasattr(batch, "bck_actions") and hasattr(ctx, "bck_action_type_order")
        else []
    ):
        # print(actions )
        # print(atypes)
        mask_cat = GraphActionCategorical(
            batch,
            [model._action_type_to_mask(t, batch) for t in atypes],
            [model._action_type_to_key[t] for t in atypes],
            [None for _ in atypes],
        )
        masked_action_is_used = (1 - mask_cat.log_prob(actions, logprobs=mask_cat.logits))* (1 - batch.is_sink)
        num_trajs = len(trajs)
        batch_idx = torch.arange(num_trajs, device=batch.x.device).repeat_interleave(batch.traj_lens)
        first_graph_idx = torch.zeros_like(batch.traj_lens)
        torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
        if masked_action_is_used.sum() != 0:
            invalid_idx = masked_action_is_used.argmax().item()
            traj_idx = batch_idx[invalid_idx].item()
            timestep = invalid_idx - first_graph_idx[traj_idx].item()
            raise ValueError("Found an action that was masked out", trajs[traj_idx]["traj"][timestep], trajs[traj_idx]["traj"])

def visualize_trajectory(traj,ctx_mol):
    mols = []
    for item in traj:
        mols.append(ctx_mol.graph_to_mol(item[0]))
    return dm.to_image(mols, mol_size=(200, 200), align=True)

def get_Hfile_paths(root):
    pattern = "*.smi.gz"
    Hfile_paths = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                Hfile_paths.append(os.path.join(path,name))
    # last_file = '/mnt/ps/home/CORP/mohit.pandey/project/files.docking.org/zinc22/2d-all/H04/H04M200.smi.gz'
    # Hfile_paths.remove(last_file)
    # Hfile_paths.append(last_file)
    print('Total Files ', len(Hfile_paths))
    return Hfile_paths

def avg_traj_len(bck_traj_batch, fwd_traj_batch ):
    avg_batch_bck_traj_len, avg_batch_fwd_traj_len, avg_batch_traj_len = 0, 0,0
    for i in range(len(bck_traj_batch)):
        avg_batch_bck_traj_len += len(bck_traj_batch[i]['bck_a'])
    avg_batch_traj_len += avg_batch_bck_traj_len

    for i in range(len(fwd_traj_batch)):
        avg_batch_fwd_traj_len += len(fwd_traj_batch[i]['bck_a'])
    avg_batch_traj_len += avg_batch_fwd_traj_len

    avg_batch_fwd_traj_len = avg_batch_fwd_traj_len/len(fwd_traj_batch)
    avg_batch_bck_traj_len = avg_batch_bck_traj_len/len(bck_traj_batch)
    avg_batch_traj_len = avg_batch_traj_len/ (len(fwd_traj_batch)+len(bck_traj_batch) )
    return avg_batch_fwd_traj_len, avg_batch_bck_traj_len, avg_batch_traj_len

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def get_bemis_murcko_scaffold(smiles):
    """
    Get the Bemis-Murcko scaffold: https://pubs.acs.org/doi/10.1021/jm9602928 of a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return ""
    else:
        return ""

class DiversityFilter:
    """
    Implements Diversity Filter as described in the paper: 
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0
    """
    def __init__(
        self, 
        bucket_size = 10
    ):
        # Track the number of times a given Bemis-Murcko scaffold has been generated
        self.bucket_history = dict()
        self.bucket_size = bucket_size

    def update(
        self,
        smiles
    ):
        """
        Update the bucket history based on the sampled (or hallucinated) batch of SMILES.
        """
        # Get the Bemis-Murcko scaffold for each SMILES
        scaffolds = [get_bemis_murcko_scaffold(smiles) for smiles in smiles]
        for scaf in scaffolds:
            if scaf in self.bucket_history:
                self.bucket_history[scaf] += 1
            else:
                self.bucket_history[scaf] = 1

    def penalize_reward(
        self,
        smiles,
        rewards
    ):
        """
        Penalize sampled (or hallucinated) SMILES based on the bucket history.
        """
        # If a given scaffold has been generated more than the bucket size, truncate the reward to 0.0
        if len(smiles) > 0:
            scaffolds = [get_bemis_murcko_scaffold(s) for s in smiles]
            penalized_rewards = []
            for idx, scaf in enumerate(scaffolds):
                if scaf in self.bucket_history and self.bucket_history[scaf] > self.bucket_size:
                    penalized_rewards.append(torch.tensor(0.0+np.finfo(float).eps).unsqueeze(dim=-1))
                else:
                    penalized_rewards.append(rewards[idx])
            return torch.stack(penalized_rewards) #stacked_pen_rew
        else:
            return np.array([])
        
