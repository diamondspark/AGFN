from trajsampling import Trajectory_Sampling
from copy import deepcopy
import torch
from utils import helpers

class TrajectoryMixer():
    def __init__(self, pretrainer, num_online,num_offline) -> None:
        self.hps = pretrainer.hps
        self.traj_sampler = Trajectory_Sampling(pretrainer)
        self.num_online = num_online
        self.num_offline = num_offline

    def zinc_mix_trajectories(self, smiles_batch, model, dev):
        '''
        Returns a batch of gfn_batch_size trajectories to train GFN by mixing offine and online trajectories
        in mix_ratio and an updated replay_buffer. 
        Randomly samples 'num_online' trajectories from replay_buffer. Remainder of the batch is formed of 
        backward trajectories by sampling current P_B. 
        Replenishes the buffer with 'num_online' trajectories sampled from current P_F.
        
        TODO: Currently, off-policy training is only implemented for P_F. Backward trajectories are always 
        sampled from current P_B. Perhaps have a seperate replay buffer to save old bck tejctories in that. 
        '''
                
        _,cur_result = self.traj_sampler.sample_tau_from_PF(model,dev,n_trajs= self.num_online)

        if len(smiles_batch)>0:
            # print(smiles_batch)
            data,flipped_data, offln_reward_tuple = self.traj_sampler.sample_tau_from_PB(smiles_batch, model, dev)
            # print('trajmixing.py offln_rew_tup', offln_reward_tuple[0], offln_reward_tuple[1], offln_reward_tuple[2])
        # print('TrajMixing.py sample_tau_from_PB success', )
        avg_batch_fwd_traj_len, avg_batch_bck_traj_len, avg_batch_traj_len = helpers.avg_traj_len(flipped_data, cur_result )

        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # print(self._wid, 'traj mixing gfn_batch reverse smiles ', )
        gfn_batch = cur_result+flipped_data

        # rb_result : online_trajs
        # flipped_data : offline_trajs
        # print(rb_result)
        # return gfn_batch , [avg_batch_fwd_traj_len, avg_batch_bck_traj_len, avg_batch_traj_len], offln_reward_tuple
        return [cur_result, flipped_data], [avg_batch_fwd_traj_len, avg_batch_bck_traj_len, avg_batch_traj_len], offln_reward_tuple