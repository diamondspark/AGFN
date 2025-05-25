
import torch
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel
import pandas as pd
import numpy as np
import pickle
from agfn.reward import RewardFineTune
from agfn.gfntrainer import GFNTrainerRTB, GFNTrainer
from gflownet.models.graph_transformer import PrunedGraphTransformerGFN, PrunedGraphTransformerGFNPrior, mlp
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionCategorical
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.graph_sampling import GraphSampler
from gflownet.models.conditionals import ConditionalInfo
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
import torch_geometric.data as gd
from rdkit import Chem

class FineTunerRTB():
    def __init__(self, hps, conditional_range_dict, cond_prop_var, load_path,  rank, world_size, ft_conditionals_dict=None) -> None:

        self.device = torch.device(f'cuda:{rank}') if rank is not None else torch.device('cpu')
        self.rank = rank
        self.allow_charge = False
        self.legal_atoms = set(hps['atoms'])
        self.tasks = [
            'Caco2',
            'LD50',
            'Lipophilicity',
            'Solubility',
            'Solubility',
            'BindingRate',
            'MicroClearance',
            'HepatocyteClearance'
        ]
        if hps['offline_data']:
            self.task_df = pd.read_csv(hps.offline_df_path)
            self.valid_smiles = self.preprocess(self.task_df)
            self.widx2smiles = np.array_split(self.valid_smiles, hps['num_workers'])

        if hps['task'] in self.tasks:
            with open(hps.task_model_path, 'rb') as f:
                self.task_model, self.Y_scaler =  pickle.load(f)                
        else:
            self.task_model, self.Y_scaler = None, None

        self.rng = np.random.default_rng(hps['random_seed'])
        self.ckpt_freq = hps['checkpoint_every']
        # num_cond_dim here determines the size of input layer of the MLP in GraphTransformer.c2h()
        # num_cond_dim is num_thermometer_dim (say 16) for lower and upper bound for each molecular property 
        # concatenated together.
        self.ctx =  MolBuildingEnvContext(num_cond_dim=2*hps["num_thermometer_dim"]*(len(conditional_range_dict)-1), #chiral_types=[ChiralType.CHI_UNSPECIFIED],
                                          charges=hps.get('charges',[0]), 
                                          atoms = hps.get('atoms',["C", "N", "O", "F", "P", "S"]), 
                                          num_rw_feat=0) # 2*hps["num_thermometer_dim"]*len(conditional_range_dict); -2 for 'zinc_radius', 'qed
        self.ctx.num_rw_feat = 0
        self.env = GraphBuildingEnv()
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls
        self.algo = TrajectoryBalance(self.env,self.ctx,self.rng,hps) 

        self.graph_sampler = GraphSampler(self.ctx,self.env, hps["max_traj_len"],hps["max_nodes"],self.rng,hps["sample_temp"],
                                correct_idempotent=False, pad_with_terminal_state=True)
        self.cond_info_task = ConditionalInfo(conditional_range_dict, cond_prop_var, hps['num_thermometer_dim'], hps['OOB_percent'])
        self.reward = RewardFineTune(conditional_range_dict,ft_conditionals_dict, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'], hps)

        self.gfn_trainer = GFNTrainerRTB(hps, self.algo, self.rng, self.device, self.env, self.ctx) 

        loaded_dict = torch.load(load_path,map_location='cpu')
        self.pretrain_hps, self.start_step, self.opt = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt']
        
        model_dict = loaded_dict['models_state_dict'][0]
        tmp_flag = False
        for key in model_dict:
            if 'module' in key:
                new_model_dict= {}
                for k in model_dict:
                    val = model_dict[k]
                    k_new = k.replace('module.','')
                    new_model_dict[k_new] = val
                self.gfn_trainer.model.load_state_dict(new_model_dict)
                self.gfn_trainer.model_prior.load_state_dict(new_model_dict)
                tmp_flag = True
                break
        if not tmp_flag:
            print(self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0]))
            print("len loaded dict:", len(loaded_dict['models_state_dict'][0]))
            self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0]) 
            self.gfn_trainer.model_prior.load_state_dict(loaded_dict['models_state_dict'][0]) 
        
        print("len loaded dict:", len(loaded_dict['models_state_dict'][0]))
        pruned_model = PrunedGraphTransformerGFN(full_model=self.gfn_trainer.model,ctx=self.ctx,hps=hps)
        self.gfn_trainer.model = pruned_model
        
        logZ_pruned_prior = PrunedGraphTransformerGFNPrior(full_model=self.gfn_trainer.model_prior,ctx=self.ctx,hps=hps)
        self.gfn_trainer.model_prior = logZ_pruned_prior
        
        
        if world_size > 1:
            self.gfn_trainer.model = DistributedDataParallel(self.gfn_trainer.model.to(rank), device_ids=[rank], 
                                                            output_device=rank)
        else:
            self.gfn_trainer.model.to(self.device)
        
        print('finetuner start step', self.start_step)

        if hps['layerwise_lr']:
            higher_lr = hps['learning_rate']*10
            params_to_update = [
                                {"params": self.gfn_trainer.model.transf.x2h.parameters()},
                                {"params": self.gfn_trainer.model.transf.e2h.parameters()},
                                {"params": self.gfn_trainer.model.transf.graph2emb.parameters()},
                                {"params": self.gfn_trainer.model.transf.c2h.parameters(), "lr": higher_lr},
                                {"params": self.gfn_trainer.model.mlps.parameters(), "lr":higher_lr},
                                {"params": self.gfn_trainer.model.logZ.parameters()},
                               ]
            self.gfn_trainer.opt = torch.optim.Adam(
                                        params_to_update,
                                        hps['learning_rate'],
                                        (hps["momentum"], 0.999),
                                        weight_decay=hps["weight_decay"],
                                        eps=hps["adam_eps"],
                                        )

        
        if hps['perturb_logZ']:
            # Add small noise to the weights of the logZ module
            parameters = self.gfn_trainer.model.module.logZ.parameters() if world_size>1 else self.gfn_trainer.model.logZ.parameters()
            for param in parameters:
                param.data += torch.randn_like(param) * 0.01  # Adding Gaussian noise with stddev 0.01

        if world_size>1:
            self.gfn_trainer.model = DistributedDataParallel(self.gfn_trainer.model.to(rank), device_ids=[rank], 
                                                                output_device=rank)
        else:
            self.gfn_trainer.model = self.gfn_trainer.model.to(rank)
        self.gfn_trainer.model_prior = self.gfn_trainer.model_prior.to(rank)
        
        self.hps = hps

    def preprocess(self, df):
        mols, valid_smiles = [], []
        for smiles in df['SMILES']:
            if self.allow_charge == False:
                if ('+' in smiles) or ('-' in smiles) or ('.' in smiles):
                    continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol_is_valid = False
            else:
                mol_is_valid = True
                atoms = mol.GetAtoms()
                for atom in atoms:
                    if not atom.GetSymbol() in self.legal_atoms:
                        mol_is_valid = False
                        break
            if mol_is_valid:
                mols.append(mol)
                valid_smiles.append(smiles)
        return valid_smiles

    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if send_to_device:
            obj.to(self.device)
        if self.hps['num_workers'] > 0 and obj is not None:
            placeholder = mp_object_wrapper(
                obj,
                self.hps['num_workers'],
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=False,
            )
            return placeholder, torch.device("cpu")
        else:
            return obj, self.device
        

class FineTuner():
    def __init__(self, hps, conditional_range_dict, cond_prop_var, load_path,  rank, world_size, ft_conditionals_dict=None) -> None:
        
        self.device = torch.device(f'cuda:{rank}') if rank is not None else torch.device('cpu')
        self.rank = rank
        self.allow_charge = False
        self.legal_atoms = set(hps['atoms'])
        self.tasks = [
            'Caco2',
            'LD50',
            'Lipophilicity',
            'Solubility',
            'Solubility',
            'BindingRate',
            'MicroClearance',
            'HepatocyteClearance'
        ]
        if hps.offline_data:
            self.task_df = pd.read_csv(hps.offline_df_path)
            self.valid_smiles = self.preprocess(self.task_df)
            self.widx2smiles = np.array_split(self.valid_smiles, hps['num_workers'])

        if hps['task'] in self.tasks:
            with open(hps.task_model_path, 'rb') as f:
                self.task_model, self.Y_scaler = pickle.load(f)
        else:
            self.task_model, self.Y_scaler = None, None

        self.rng = np.random.default_rng(hps['random_seed'])
        self.ckpt_freq = hps['checkpoint_every']
        # num_cond_dim here determines the size of input layer of the MLP in GraphTransformer.c2h()
        # num_cond_dim is num_thermometer_dim (say 16) for lower and upper bound for each molecular property 
        # concatenated together.
        self.ctx = MolBuildingEnvContext(num_cond_dim=2*hps["num_thermometer_dim"]*(len(conditional_range_dict)-1), #chiral_types=[ChiralType.CHI_UNSPECIFIED],
                                          charges=hps.get('charges',[0]), 
                                          atoms = hps.get('atoms',["C", "N", "O", "F", "P", "S"]), 
                                          num_rw_feat=0) # 2*hps["num_thermometer_dim"]*len(conditional_range_dict); -2 for 'zinc_radius', 'qed
    
        self.ctx.num_rw_feat = 0
        self.env = GraphBuildingEnv()
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls
            
        self.algo = TrajectoryBalance(self.env,self.ctx,self.rng,hps) 

        self.graph_sampler = GraphSampler(self.ctx,self.env, hps["max_traj_len"],hps["max_nodes"],self.rng,hps["sample_temp"],
                                correct_idempotent=False, pad_with_terminal_state=True)
        self.cond_info_task = ConditionalInfo(conditional_range_dict, cond_prop_var, hps['num_thermometer_dim'], hps['OOB_percent'])

        self.reward = RewardFineTune(conditional_range_dict,ft_conditionals_dict, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'], hps)

        # self.gfn_trainer = SEHFragTrainer(hps,self.device)
        self.gfn_trainer = GFNTrainer(hps, self.algo, self.rng, self.device, self.env, self.ctx) 

        loaded_dict = torch.load(load_path,map_location='cpu')
        self.pretrain_hps, self.start_step, self.opt = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt']

        model_dict = loaded_dict['models_state_dict'][0]
        tmp_flag = False
        for key in model_dict:
            if 'module' in key:
                new_model_dict= {}
                for k in model_dict:
                    val = model_dict[k]
                    k_new = k.replace('module.','')
                    new_model_dict[k_new] = val
                self.gfn_trainer.model.load_state_dict(new_model_dict)
                tmp_flag = True
                break
        if not tmp_flag:
            print(self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0]))
            print("len loaded dict:", len(loaded_dict['models_state_dict'][0]))
            self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0]) # this line was accidentally turned off in some earlier experiments, hence even this code trained from scratch. Eg. golder shadow 103 and prior expts.

        print('finetuner start step', self.start_step)

        #TODO : make this model.module.layer=....
        if hps['reset_wts']:
            new_layers = dict()
            new_layers['stop'] =  nn.Sequential(nn.Linear(in_features=256, out_features=1, bias=True))
            new_layers['add_node'] = nn.Sequential(nn.Linear(in_features=128, out_features=6, bias=True))
            new_layers['set_node_attr'] = nn.Sequential(nn.Linear(in_features=128, out_features=6, bias=True))
            new_layers['add_edge']= nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True))
            new_layers['set_edge_attr'] = nn.Sequential(nn.Linear(in_features=128, out_features=2, bias=True))
            new_layers['remove_node'] = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True))
            new_layers['remove_node_attr'] = nn.Sequential(nn.Linear(in_features=128, out_features=4, bias=True))
            new_layers['remove_edge'] = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True))
            new_layers['remove_edge_attr'] = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True))
            self.gfn_trainer.model.mlps = nn.ModuleDict(new_layers)
       
        if hps['reset_logZ']:
            self.gfn_trainer.model.logZ = mlp(self.ctx.num_cond_dim, hps['num_emb'] * 2, 1, 2)

        if hps['freeze_model']:
            self.gfn_trainer.model.eval()
            for name, param in self.gfn_trainer.model.named_parameters():
                param.requires_grad = False

            # # Unfreeze action MLPs and logZ and c2h
            for param in self.gfn_trainer.model.logZ.parameters():
                param.requires_grad = True

            for param in self.gfn_trainer.model.mlps.parameters():
                param.requires_grad = True

            for param in self.gfn_trainer.model.transf.c2h.parameters():
                param.requires_grad = True

            params_to_update = [p for p in self.gfn_trainer.model.parameters() if p.requires_grad]
            self.gfn_trainer.opt = torch.optim.Adam(
                                        params_to_update,
                                        hps["learning_rate"],
                                        (hps["momentum"], 0.999),
                                        weight_decay=hps["weight_decay"],
                                        eps=hps["adam_eps"],
                                        )

        if hps['layerwise_lr']:
            higher_lr = hps['learning_rate']*10
            params_to_update = [
                                {"params": self.gfn_trainer.model.transf.x2h.parameters()},
                                {"params": self.gfn_trainer.model.transf.e2h.parameters()},
                                {"params": self.gfn_trainer.model.transf.graph2emb.parameters()},
                                {"params": self.gfn_trainer.model.transf.c2h.parameters(), "lr": higher_lr},
                                {"params": self.gfn_trainer.model.mlps.parameters(), "lr":higher_lr},
                                {"params": self.gfn_trainer.model.logZ.parameters()},
                               ]
            self.gfn_trainer.opt = torch.optim.Adam(
                                        params_to_update,
                                        hps['learning_rate'],
                                        (hps["momentum"], 0.999),
                                        weight_decay=hps["weight_decay"],
                                        eps=hps["adam_eps"],
                                        )

        if world_size > 1:
            self.gfn_trainer.model = DistributedDataParallel(self.gfn_trainer.model.to(rank), device_ids=[rank], 
                                                            output_device=rank)#, find_unused_parameters=True)
        else:
            self.gfn_trainer.model.to(self.device)

        if hps['perturb_logZ']:
            # Add small noise to the weights of the logZ module
            if world_size>1:
                for param in self.gfn_trainer.model.module.logZ.parameters():
                    param.data += torch.randn_like(param) * 0.01
            else:
                for param in self.gfn_trainer.model.logZ.parameters():
                    param.data += torch.randn_like(param) * 0.01
                
        self.hps = hps

    def preprocess(self, df):
        mols, valid_smiles = [], []
        for smiles in df['SMILES']:
            if self.allow_charge == False:
                if ('+' in smiles) or ('-' in smiles) or ('.' in smiles):
                    continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol_is_valid = False
            else:
                mol_is_valid = True
                atoms = mol.GetAtoms()
                for atom in atoms:
                    if not atom.GetSymbol() in self.legal_atoms:
                        mol_is_valid = False
                        break
            if mol_is_valid:
                mols.append(mol)
                valid_smiles.append(smiles)
        return valid_smiles

    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        # print('obj in wrap_for_mp', obj)
        if send_to_device:
            obj.to(self.device)
        if self.hps['num_workers'] > 0 and obj is not None:
            placeholder = mp_object_wrapper(
                obj,
                self.hps['num_workers'],
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=False,
            )
            return placeholder, torch.device("cpu")
        else:
            return obj, self.device
        
 