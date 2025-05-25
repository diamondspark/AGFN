
import torch
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel
import pandas as pd
import numpy as np
import pickle
from agfn.reward import RewardFineTune
from agfn.gfntrainer import GFNTrainer
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionCategorical
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.graph_sampling import GraphSampler
from gflownet.models.conditionals import ConditionalInfo
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
import torch_geometric.data as gd

class TaskTrainer():
    def __init__(self, hps, conditional_range_dict, cond_prop_var,  rank, world_size, task_conditionals_dict= None) -> None:

        self.device = torch.device(f'cuda:{rank}') if rank is not None else torch.device('cpu')
        self.rank = rank
        self.allow_charge = False
        self.legal_atoms = set(["C", "N", "O", "F", "P", "S"])
        
        
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
                                          num_rw_feat=0)
        self.ctx.num_rw_feat = 0
        self.env = GraphBuildingEnv()
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls

        self.algo = TrajectoryBalance(self.env,self.ctx,self.rng,hps) 
        self.graph_sampler = GraphSampler(self.ctx,self.env, hps["max_traj_len"],hps["max_nodes"],self.rng,hps["sample_temp"],
                                correct_idempotent=False, pad_with_terminal_state=True)
        self.cond_info_task = ConditionalInfo(conditional_range_dict, cond_prop_var, hps['num_thermometer_dim'], hps['OOB_percent'])
        # self.reward = Reward(conditional_range_dict, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'])
        self.reward = RewardFineTune(conditional_range_dict,task_conditionals_dict, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'], hps)
        # self.gfn_trainer = SEHFragTrainer(hps,self.device)
        self.gfn_trainer = GFNTrainer(hps, self.algo, self.rng, self.device, self.env, self.ctx) 

        if hps.task_conditionals:
            self.gfn_trainer.ctx.num_ft_cond_dim = 2*hps["num_thermometer_dim"]*(len(task_conditionals_dict))
            self.gfn_trainer.model = GraphTransformerGFNFineTune(self.ctx, num_emb=hps["num_emb"], num_layers=hps["num_layers"],
                                         do_bck=True, num_graph_out=int(hps["tb_do_subtb"]), num_mlp_layers=hps['num_mlp_layers'])

        if world_size > 1:
            self.gfn_trainer.model = DistributedDataParallel(self.gfn_trainer.model.to(rank), device_ids=[rank], 
                                                            output_device=rank)#, find_unused_parameters=True)
        else:
            self.gfn_trainer.model.to(self.device)

        self.hps = hps
        

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
        
