import numpy as np
import torch
from agfn.gfntrainer import GFNTrainer
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionCategorical
import torch_geometric.data as gd
from gflownet.algo.graph_sampling import GraphSampler
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
from gflownet.models.conditionals import ConditionalInfo
from pretrainSrc.reward_2d import Reward
import math
# from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import DataParallel
from rdkit.Chem.rdchem import BondType, ChiralType


class SplittingDataParallel(DataParallel):
    def scatter(self, data_list, devices):
        return [i.to(f"cuda:{j}") for i, j in zip(data_list, devices)]

    def gather(self, outputs, output_device):
        outputs = [i for i in outputs if i[0] is not None]
        gathered_outputs = []
        for i in range(max(map(len, outputs))):
            if isinstance(outputs[0][i], GraphActionCategorical):
                gathered_outputs.append(
                    GraphActionCategorical.stack([outputs[j][i].to(output_device) for j in range(len(outputs))])
                )
            else:
                gathered_outputs.append(
                    torch.cat([outputs[j][i].to(output_device) for j in range(len(outputs))], dim=0)
                )
        return gathered_outputs


class Pretrainer():
    def __init__(self, hps, conditional_range_dict, cond_prop_var, rank=None, load_path=None, world_size=1) -> None:
        self.device = torch.device(f'cuda:{rank}') if rank is not None else torch.device('cpu')
        self.rank = rank
        self.world_size = world_size
        self.hps = hps
        self.rng = np.random.default_rng(hps['random_seed'])
        self.ckpt_freq = hps['checkpoint_every']
        # num_cond_dim here determines the size of input layer of the MLP in GraphTransformer.c2h()
        # num_cond_dim is num_thermometer_dim (say 16) for lower and upper bound for each molecular property 
        # concatenated together.
        self.ctx =  MolBuildingEnvContext(num_cond_dim=2*hps["num_thermometer_dim"]*(len(conditional_range_dict)-1), #chiral_types=[ChiralType.CHI_UNSPECIFIED],
                                          charges=hps.get('charges',[0]), 
                                          atoms = hps.get('atoms',["C", "N", "O", "F", "P", "S"]),
                                          num_rw_feat=0) # 2*hps["num_thermometer_dim"]*len(conditional_range_dict); -2 for 'zinc_radius', 'qed
        print('pretrainer.pyy num node dim ctx ',self.ctx.num_node_dim)
        print('pretrainer.py num node ctx characterization',self.ctx.atom_attr_values)
        self.env = GraphBuildingEnv()
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls
        self.algo = TrajectoryBalance(self.env,self.ctx,self.rng,hps) 
        self.graph_sampler = GraphSampler(self.ctx,self.env, hps["max_traj_len"],hps["max_nodes"],self.rng,hps["sample_temp"],
                                correct_idempotent=False, pad_with_terminal_state=True)
        self.gfn_trainer = GFNTrainer(hps, self.algo, self.rng, self.device, self.env, self.ctx)

        if 0:
            self.gfn_trainer.model = SplittingDataParallel(
                self.gfn_trainer.model, self._multi_gpu_devices, follow_batch=["edge_index", "non_edge_index"]
            )
            import pdb; pdb.set_trace()


        if world_size > 1:
            # os.environ['MASTER_ADDR'] = 'localhost'
            # os.environ['MASTER_PORT'] = '12378'
            # dist.init_process_group('nccl', rank=rank, world_size=world_size)
            self.gfn_trainer.model = DistributedDataParallel(self.gfn_trainer.model.to(rank), device_ids=[rank], 
                                                            output_device=rank)#, find_unused_parameters=True)
        else:
            self.gfn_trainer.model.to(self.device)
        trainable_params = sum(p.numel() for p in self.gfn_trainer.model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.gfn_trainer.model.parameters() if not p.requires_grad)
        print('pretrainer.py trainable_params ', trainable_params, 'non_trainavle params ', non_trainable_params)
        # if isinstance(self.gfn_trainer.model, torch.nn.parallel.DistributedDataParallel):

        if hps['load_saved_model']:
            loaded_dict = torch.load(load_path,map_location='cpu')
            # print(loaded_dict)
            _, self.start_step, self.opt = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt']
            # self.hps, self.start_step, self.opt, self.opt_Z = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt'], loaded_dict['opt_Z']
            self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0])
            print('pretrainer start step', self.start_step)

        # if hps['load_saved_model']:
        #     self.gfn_trainer.model = nn.parallel.DistributedDataParallel(self.gfn_trainer.model)
        #     self.gfn_trainer.model.to(self.device)
        #     loaded_dict = torch.load(load_path,map_location='cpu')
        #     # print(loaded_dict)
        #     self.hps, self.start_step, self.opt = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt']
        #     # self.hps, self.start_step, self.opt, self.opt_Z = loaded_dict['hps'], loaded_dict['step'], loaded_dict['opt'], loaded_dict['opt_Z']
        #     print("pretrainer.py gfntrainer model ", self.gfn_trainer.model )
        #     print("pretrainer.py loaded dict ", loaded_dict['models_state_dict'])
        #     self.gfn_trainer.model.load_state_dict(loaded_dict['models_state_dict'][0])
        #     print('pretrainer start step', self.start_step)
        else:
            self.start_step = 0
            self.hps = hps
        
        # print('pretrainer.py gfntrainer model',self.gfn_trainer.model)
        self.task = ConditionalInfo(conditional_range_dict, cond_prop_var, hps['num_thermometer_dim'], hps['OOB_percent'])
        self.reward = Reward(conditional_range_dict, cond_prop_var, self.hps["reward_aggergation"], self.hps['atomenv_dictionary'], self.hps['zinc_rad_scale'], self.hps)

    def mixing_counts(self,gfn_batch_size,mix_ratio=0.5):
        # task = self.task
        num_online = math.floor(gfn_batch_size*mix_ratio)
        num_offline = gfn_batch_size - num_online
        print(f"Sampling {num_online} trajectories from current policy \n\
              Sampling {num_offline} trajectories from offline data")
        # cond_info = task.sample_conditional_information(num_offline + num_online, 0) 
        self.num_online, self.num_offline = num_online, num_offline
        # self.cond_info = cond_info
        return num_online, num_offline#, cond_info
        
        
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
                pickle_messages=True,
            )
            return placeholder, torch.device("cpu")
        else:
            return obj, self.device

    # def build_task(self):
    #     self.task = SEHTask(
    #                 dataset=[],
    #                 temperature_distribution=self.hps["temperature_sample_dist"],
    #                 temperature_parameters=self.hps["temperature_dist_params"],
    #                 rng=self.rng,
    #                 num_thermometer_dim=self.hps["num_thermometer_dim"],
    #                 wrap_model=self._wrap_for_mp,
    #             )
    #     # self.task.models['seh'].to(self.device)


    def build_training_dataloader():
        pass

