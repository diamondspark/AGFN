import pickle
import numpy as np
import json
from rdkit import Chem
import subprocess
from agfn.reward import Reward

class RewardDockFineTune(Reward):
    def __init__(self, cond_range_dict, ft_cond_dict,cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps,gfn_samples_path ) -> None:
        super().__init__(cond_range_dict, cond_prop_var, reward_aggregation, molenv_dict_path, zinc_rad_scale, hps)
        self.hps = hps
        self.gfn_samples_path = gfn_samples_path
        self.vina_path = hps['vina_path']
        with open(f'./data/docking/tmp_config.json', "w") as f:
            json.dump(hps.target_grid, f, indent=4)

    def vina_docking_reward(self, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols] 
        outs = self.vina.calculate_rewards(smiles_list)
        return outs
    
    def task_reward(self, task, mols):
        dock_config = {"5ht1b": {
        "receptor": "/groups/cherkasvgrp/Student_backup/mkpandey/gfn_pretrain_test_env_public/data/docking/5ht1b.pdbqt", #os.path.join(DATA_DIR, "5ht1b/5ht1b.pdbqt"),
        "center_x": -26.602,
        "center_y": 5.277,
        "center_z": 17.898,
        "size_x": 22.5,
        "size_y": 22.5,
        "size_z": 22.5,
    }}
        
        vina_docking_cmd = ["python", "./src/apps/docking/gpuvina.py", self.hps.target_name, self.gfn_samples_path, self.vina_path]
        subprocess.run(vina_docking_cmd, check=True)
        with open(f'{self.gfn_samples_path}/{self.hps.target_name}_docked.pkl','rb') as f:
            outs = pickle.load(f)
        true_task_score = np.array(outs[1]) 
        flat_rewards_task = np.expand_dims(np.array(np.array(outs[2])),axis=1)
        return flat_rewards_task, true_task_score  