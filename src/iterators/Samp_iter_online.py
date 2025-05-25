import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader, IterableDataset
from rdkit.Chem.Scaffolds import MurckoScaffold
import time
class Sampling_Iterator_Online(IterableDataset):
    def __init__(self, ft_trainer, wrapped_model, dev, num_online, beta ):#ctx, hps,traj_sampler):
        self.num_online = num_online
        self.cond_info_task = ft_trainer.task
        self.wrapped_model = wrapped_model
        self.dev = dev
        self.beta = beta
        self.hps = ft_trainer.hps
        self.graph_sampler = ft_trainer.graph_sampler
        # self.task = ft_trainer.task
        self.ctx = ft_trainer.ctx
        self.algo = ft_trainer.algo
        self.reward = ft_trainer.reward
        self.sub_batch_size =  ft_trainer.hps['training_batch_size']
        # if self.hps.get('seed_scaffold',None) or self.hps.get('seed_smiles',None):
        #     if self.hps['seed_scaffold']:
        #         self.seed_mol = Chem.MolFromSmiles(self.hps['seed_smiles'])
        #         scaffold = MurckoScaffold.GetScaffoldForMol(self.seed_mol)
        #         self.seed_graph = self.ctx.mol_to_graph(scaffold)
        #     else:
        #         self.seed_mol = Chem.MolFromSmiles(self.hps['seed_smiles'])
        #         self.seed_graph = self.ctx.mol_to_graph(self.seed_mol)
        # else:
        #     self.seed_mol, self.seed_graph = None, None
        self.seed_mol, self.seed_graph = None, None
        self.reward.seed_mol=self.seed_mol


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        for j in range(self.hps['num_iter']):
            try:
                cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)

                # traj_sampler.sample_tau_from_PF(pretrainer.gfn_trainer.model,torch.device(f'cuda:{rank}'),n_trajs=n_trajs)
                with torch.no_grad():
                    online_trajs_sampled = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                            cond_info_encoding.to(self.dev),
                                                                            self.dev, self.hps['random_action_prob'], seed_graph=self.seed_graph )
                # print(f'sampiterol.py {len(online_trajs_sampled)} sampled')
                for j in range(0, self.num_online, self.sub_batch_size):
                    online_trajs = online_trajs_sampled[j:j+self.sub_batch_size]
                    avg_batch_len, avg_fwd_logprob, avg_bck_logprob = 0, 0,0
                    for i in range(len(online_trajs)):
                        avg_batch_len += len(online_trajs[i]['bck_a'])
                        avg_fwd_logprob += online_trajs[i]['fwd_logprob'][0]
                        avg_bck_logprob += online_trajs[i]['bck_logprob'][0]

                    valid_idcs = torch.tensor([i + 0 for i in range(len(online_trajs)) if (online_trajs[i + 0]["is_valid"])]).long()
                    if len(valid_idcs)==0:
                        with open(self.hps["log_dir"] + f"/invalid_mols{j}.txt", "w") as f:
                            content = f"{str(online_trajs)}"
                            f.write(content)
                    
                    mols = [self.ctx.graph_to_mol(online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
                    rew_tup = self.reward.molecular_rewards(mols)
                    online_rew = torch.Tensor(rew_tup[2]).unsqueeze(dim=1)
                    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
                    # print(smiles_list)
                    pred_reward_online = torch.zeros(len(online_trajs), online_rew.shape[1])
                    pred_reward_online[valid_idcs - 0] = online_rew
                    pred_reward = pred_reward_online

                    #DOCKING TASK REWARDS ONLINE
                    # t0 = time.time()
                    # flat_task_reward_online, docking_scores_on, sim_on = self.reward.unidocking_reward(mols, self.seed_mol)
                    # flat_task_reward_online, docking_scores_on, sim_on = self.reward.docking_reward(mols, self.seed_mol)
                    
                    # print(f'sampiterol.py docking reward for {len(mols)} took {time.time()-t0} secs')
                    # flat_rewards = rew*torch.Tensor(flat_task_reward_online).unsqueeze(axis=1)
                    # print('flat_reards ', flat_rewards.shape)
                    # flat_rewards_qed = torch.unsqueeze(torch.Tensor(np.array([QED.qed(mol) for mol in mols])), dim=1) #QED reward
                    # flat_rewards =  rew*flat_rewards_task  #torch.mul(rew,flat_rewards_task) #flat_rewards_qed
                    # print ('ft_new.py flat_Rewards qed ', flat_rewards.shape)
                    # print ('ft_new.py flat_Rewards seh ', flat_rewards_seh.shape)
                    # flat_rewards = softplus(flat_rewards)
                    # flat_rewards = torch.unsqueeze(torch.Tensor(np.array([Crippen.MolLogP(mol) for mol in mols])), dim=1) #reward_2d.logP(mols)
                    # flat_rewards = torch.unsqueeze(torch.Tensor(np.array([CalcFractionCSP3(mol) for mol in mols])), dim=1) #reward_2d.fsp3(mols)
                    #Drug likeliness score
                    
                    # print('sampled smiles_list ',smiles_list)
                    # flat_rewards = torch.unsqueeze(torch.Tensor(np.array([self.drug_like_model.test(smiles) for smiles in smiles_list])), dim=1)
                
                    # print('flat_reards ', flat_rewards.shape)
                    # #Ensure reward is set to 0 for invalid trajectories (molecules)
                    # pred_reward = torch.zeros((len(online_trajs), flat_rewards.shape[1]))
                    # pred_reward[valid_idcs - 0] = online_rew #flat_rewards.float()

                    # beta_dict = {'beta':self.beta*torch.ones(self.num_online)}
                    # log_rewards = self.task.cond_info_to_logreward(beta_dict, pred_reward)
                
                    beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                    log_rewards = self.beta_to_logreward(beta_vector, pred_reward)

                    gfn_batch = self.algo.construct_batch(online_trajs, cond_info_encoding, log_rewards)#pred_reward.squeeze(dim=1))
                    
                    
                    # gfn_batch.num_sub_online_trajs = len(sub_online_trajs)
                    # gfn_batch.num_sub_offline_trajs = len(sub_offline_trajs)
                    gfn_batch.num_online = gfn_batch.num_sub_online_trajs = len(online_trajs)
                    gfn_batch.num_offline = 0
                    gfn_batch.flat_rewards = pred_reward
                    gfn_batch.online_flat_rewards = pred_reward
                    gfn_batch.offline_flat_rewards = torch.Tensor(0)
                    # print('sampiterinmem.py avg offline rew ',  torch.mean(gfn_batch.offline_flat_rewards).item())
                    # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
                    gfn_batch.valid_percent = len(valid_idcs)/len(online_trajs)
                    gfn_batch.offln_zinc_rad = 0
                    gfn_batch.onln_zinc_rad = np.average(rew_tup[3])
                    gfn_batch.offline_avg_tpsa =  0
                    gfn_batch.online_avg_tpsa =  np.average(rew_tup[1][0][0])
                    gfn_batch.offln_num_rings = 0
                    gfn_batch.online_num_rings = np.average(rew_tup[1][1][0])
                    gfn_batch.offline_avg_SAS = 0
                    gfn_batch.online_avg_SAS =  np.average(rew_tup[1][2][0])
                    gfn_batch.offline_qed = 0 
                    gfn_batch.online_qed = np.average(rew_tup[1][-1][0])                            
                    yield gfn_batch, [avg_batch_len,0, avg_batch_len], None, mols, []
            except Exception as e:
                print(e)
                raise(e)
                continue
    def beta_to_logreward(self,beta_vector, pred_reward):
        scalar_logreward = pred_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            beta_vector.shape
        ), f"dangerous shape misatch: {scalar_logreward.shape} vs {beta_vector.shape}"
        return scalar_logreward * beta_vector   # log(r(x)**beta) = beta*log(r(x))


def build_train_loader(hps,finetuner):
    wrapped_model, dev = finetuner._wrap_for_mp(finetuner.gfn_trainer.model, send_to_device=True)
    iterator = Sampling_Iterator_Online(finetuner,wrapped_model,dev, hps['sampling_batch_size'], hps['beta_exp'])
    train_loader = DataLoader(iterator,batch_size =None, num_workers = hps['num_workers'])
    return train_loader