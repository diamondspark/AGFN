import torch
from torch.utils.data import DataLoader, IterableDataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils.helpers import DiversityFilter
from agfn.backtraj import ReverseFineTune
import pickle
import numpy as np

class FTSampling_Iterator(IterableDataset):
    def __init__(self, ft_trainer, wrapped_model, wrapped_model_prior, dev, num_online, beta, ft_conditionals_dict= None ):
        self.num_online = num_online
        self.sub_batch_size =  ft_trainer.hps['training_batch_size']
        self.batch_size = ft_trainer.hps['sampling_batch_size']
        if ft_trainer.hps.offline_data:
            self.num_offline = num_online
            self.widx2smiles = ft_trainer.widx2smiles
            self.small_task_data = True if len(self.widx2smiles[0])<self.sub_batch_size else False
        self.cond_info_task = ft_trainer.cond_info_task
        self.wrapped_model = wrapped_model
        self.wrapped_model_prior = wrapped_model_prior
        self.dev = dev
        self.beta = beta
        self.gfn_samples_path = getattr(ft_trainer, 'gfn_samples_path', None)
        self.hps = ft_trainer.hps
        self.graph_sampler = ft_trainer.graph_sampler
        self.ft_conditionals_dict = ft_conditionals_dict
        self.ctx = ft_trainer.ctx
        self.algo = ft_trainer.algo
        self.reward = ft_trainer.reward
        self.offline_data = ft_trainer.hps.offline_data
        self.task_model = ft_trainer.task_model
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
        if (self.hps['task'] in self.tasks):
            self.Y_scaler = ft_trainer.Y_scaler
        self.reverse = ReverseFineTune(ft_trainer.env, self.ctx,self.hps, ft_trainer.rng)
        if 'seed_smiles' in self.hps:
            if self.hps['seed_scaffold']:
                print('trajsamp.py Optimizing ', self.hps['seed_scaffold'])
                scaffold = Chem.MolFromSmiles(self.hps['seed_scaffold'])
            else:
                print('trajsamp.py Optimizing ', self.hps['seed_smiles'])
                mol = Chem.MolFromSmiles(self.hps['seed_smiles'])
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            self.seed_graph = self.ctx.mol_to_graph(scaffold)
        else:
            self.seed_graph = None
        self.div_fil = DiversityFilter()
        
        # Then assign the dictionary
        self.task_model_reward_funcs = {
            'Caco2': self._reward_caco2,
            'LD50': self._reward_ld50,
            'Lipophilicity': self._reward_lipophilicity,
            'Solubility': self._reward_solubility,
            'BindingRate': self._reward_binding_rate,
            'MicroClearance': self._reward_micro_clearance,
            'HepatocyteClearance': self._reward_hepatocyte_clearance
        }
        
    def _reward_caco2(self, x):
        return self.reward.caco2(self.Y_scaler, self.task_model, x)

    def _reward_ld50(self, x):
        return self.reward.toxicity(self.Y_scaler, self.task_model, x)

    def _reward_lipophilicity(self, x):
        return self.reward.lipophilicity(self.Y_scaler, self.task_model, x)

    def _reward_solubility(self, x):
        return self.reward.solubility(self.Y_scaler, self.task_model, x)

    def _reward_binding_rate(self, x):
        return self.reward.binding_rate(self.Y_scaler, self.task_model, x)

    def _reward_micro_clearance(self, x):
        return self.reward.micro_clearance(self.Y_scaler, self.task_model, x)

    def _reward_hepatocyte_clearance(self, x):
        return self.reward.hepatocyte_clearance(self.Y_scaler, self.task_model, x)

    def smiles_2_offln_trajs(self, smiles_batch, wrapped_model,dev):
         mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch if smiles is not None]
         lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew = self.reward.molecular_rewards(mols)
         cond_info_bck = self.cond_info_task.compute_cond_info_backward(flat_rewards)
         self.cond_info_bck_encoding = self.cond_info_task.thermometer_encoding(cond_info_bck)
         data = self.reverse.reverse_trajectory(smiles_batch,self.cond_info_bck_encoding, wrapped_model, dev)
         flipped_data = self.reverse.flip_trajectory(data)
         return data, flipped_data, (lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        if self.offline_data:
            worker_smiles = self.widx2smiles[self._wid]
            for k in range(self.hps['num_iter']):
                self.smiles_idx = 0
                while self.smiles_idx <len(worker_smiles):
                    try:
                        cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                        cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)
                        smiles_batch = worker_smiles[self.smiles_idx:self.smiles_idx+self.batch_size]  # TODO: Account for some smiles left out at the end of the worker_smiles list
                        self.smiles_idx += self.batch_size//2  # divided by 2 because smiles_batch is batch_size/2; the remainder batch_size/2 is for online trajs
                        with torch.no_grad(): 
                            online_trajs = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                                cond_info_encoding.to(self.dev),
                                                                                self.dev, random_stop_action_prob= self.hps['random_stop_prob'],
                                                                                random_action_prob = self.hps['random_action_prob'],
                                                                                seed_graph = self.seed_graph )
                            tmp_offln_model = self.wrapped_model_prior if self.hps.type=='rtb' else self.wrapped_model
                            _, offline_trajs, offln_rew_tup = self.smiles_2_offln_trajs(smiles_batch, tmp_offln_model, self.dev) 

                        r = (len(offline_trajs)+len(online_trajs)) // self.sub_batch_size
                        for j in range(0,min(len(offline_trajs), len(online_trajs)),self.sub_batch_size):
                            sub_online_trajs = online_trajs[j : j + self.sub_batch_size]
                            avg_batch_len, avg_fwd_logprob, avg_bck_logprob = 0, 0,0
                            for i in range(len(sub_online_trajs)):
                                avg_batch_len += len(sub_online_trajs[i]['bck_a'])
                                avg_fwd_logprob += sub_online_trajs[i]['fwd_logprob'][0]
                                avg_bck_logprob += sub_online_trajs[i]['bck_logprob'][0]
                            valid_idcs = torch.tensor([i + 0 for i in range(len(sub_online_trajs)) if (sub_online_trajs[i + 0]["is_valid"]) & ("fwd_logprob" in sub_online_trajs[i+0])]).long()
                            if len(valid_idcs)==0:
                                with open(self.hps["log_dir"] + f"/invalid_mols{j}.txt", "w") as f:
                                    content = f"{str(online_trajs)}"
                                    f.write(content)
                            online_mols = [self.ctx.graph_to_mol(sub_online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
                            
                            if self.small_task_data:
                                offline_mols = [Chem.MolFromSmiles(s) for s in smiles_batch] # when offline data is small
                                online_rew_tup = self.reward.molecular_rewards(online_mols)
                                online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
                                sub_offline_trajs = offline_trajs # when offline data is small
                                offline_rew = offln_rew_tup[2] #when offline data is small 
                            else:
                                offline_mols = [Chem.MolFromSmiles(s) for s in smiles_batch[j : j + self.sub_batch_size]] # when offline data is large
                                online_rew_tup = self.reward.molecular_rewards(online_mols)
                                online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
                                sub_offline_trajs = offline_trajs[j : j + self.sub_batch_size] # when offline data is large
                                offline_rew = offln_rew_tup[2][j : j + self.sub_batch_size] # when offline data is large
                            
                            offline_rew = torch.Tensor(offline_rew).unsqueeze(dim=1)
                            if self.hps['task'] in self.task_model_reward_funcs:
                                normalized_task_rew_online, true_task_score_online = self.reward.task_reward(self.hps.task, self.task_model, online_mols)
                                normalized_task_rew_offline, true_task_score_offline = self.reward.task_reward(self.hps.task, self.task_model,
                                                                                offline_mols)
                            else:
                                normalized_task_rew_online, true_task_score_online = self.reward.task_reward(self.hps.task, self.task_model, online_mols)
                                normalized_task_rew_offline, true_task_score_offline = self.reward.task_reward(self.hps.task, self.task_model, offline_mols)
                            if self.hps.task_rewards_only:
                                flat_rewards_online, flat_rewards_offline =  torch.Tensor(normalized_task_rew_online), torch.Tensor(normalized_task_rew_offline)
                            else:
                                flat_rewards_online, flat_rewards_offline = online_rew*normalized_task_rew_online, offline_rew*normalized_task_rew_offline
                            
                            smiles_list = [Chem.MolToSmiles(mol) for mol in online_mols]
                            if self.hps['diversity_filter']:
                                self.div_fil.update(smiles_list)
                                #penalize rewards for frequently generated scaffolds
                                flat_rewards = self.div_fil.penalize_reward(smiles_list,flat_rewards_online)
                                
                            #Ensure reward is set to 0 for invalid trajectories (molecules)
                            pred_reward_online = torch.zeros(len(sub_online_trajs), flat_rewards_online.shape[1])
                            pred_reward_online[valid_idcs - 0] = flat_rewards_online.float()
                            pred_reward = torch.cat((pred_reward_online, flat_rewards_offline))
                            beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                            log_rewards = self.beta_to_logreward(beta_vector, pred_reward)
                            cond_info = torch.cat((cond_info_encoding[j : j + self.sub_batch_size],
                                                self.cond_info_bck_encoding))
                            gfn_batch = self.algo.construct_batch(sub_online_trajs+sub_offline_trajs, cond_info, log_rewards)
                            gfn_batch.num_offline = self.num_offline // r
                            gfn_batch.num_online = self.num_online // r
                            gfn_batch.flat_rewards = pred_reward.detach().cpu()
                            gfn_batch.online_flat_rewards = pred_reward_online
                            gfn_batch.offline_flat_rewards = offline_rew
                            gfn_batch.num_sub_online_trajs = len(sub_online_trajs)
                            gfn_batch.valid_percent =  len(valid_idcs)/len(sub_online_trajs)
                            gfn_batch.unique_percent = len(set(smiles_list))/len(sub_online_trajs)
                            gfn_batch.avg_batch_len = (avg_batch_len)/len(sub_online_trajs)
                            gfn_batch.avg_fwd_logprob = avg_fwd_logprob/len(sub_online_trajs)
                            gfn_batch.avg_bck_logprob = avg_bck_logprob/len(sub_online_trajs)
                            gfn_batch.offln_zinc_rad = np.average(offln_rew_tup[3])#[j : j + self.sub_batch_size])
                            gfn_batch.onln_zinc_rad = np.average(online_rew_tup[3])
                            gfn_batch.offline_avg_tpsa =  np.average(offln_rew_tup[1][0][0])
                            gfn_batch.online_avg_tpsa =  np.average(online_rew_tup[1][0][0])
                            gfn_batch.offln_num_rings = np.average(offln_rew_tup[1][1][0])
                            gfn_batch.online_num_rings = np.average(online_rew_tup[1][1][0])
                            gfn_batch.offline_avg_SAS = np.average(offln_rew_tup[1][2][0])
                            gfn_batch.online_avg_SAS =  np.average(online_rew_tup[1][2][0])
                            gfn_batch.offline_qed = np.average(offln_rew_tup[1][3][0])
                            gfn_batch.online_qed = np.average(online_rew_tup[1][3][0])
                            gfn_batch.avg_task_reward_online = np.average(normalized_task_rew_online) 
                            gfn_batch.avg_task_reward_offline = np.average(normalized_task_rew_offline)  
                            if self.hps.task is not None:
                                gfn_batch.avg_task_score_online = np.average(true_task_score_online)
                                gfn_batch.avg_task_score_offline = np.average(true_task_score_offline)
                            else:
                                gfn_batch.avg_task_reward_online, gfn_batch.avg_task_reward_offline, gfn_batch.avg_task_score_online, gfn_batch.avg_task_score_offline = 0,0,0,0

                            yield gfn_batch, online_mols

                    except Exception as e:
                        raise e
                        print(e)
                        continue
        else:
            for _ in range(self.hps['num_iter']):
                try:
                    cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                    cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)
                    with torch.no_grad():
                        online_trajs_sampled = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                                cond_info_encoding.to(self.dev),
                                                                                self.dev, random_stop_action_prob= self.hps['random_stop_prob'],
                                                                                random_action_prob = self.hps['random_action_prob'],
                                                                                seed_graph= self.seed_graph)

                    for j in range(0, self.num_online, self.sub_batch_size):
                        online_trajs = online_trajs_sampled[j:j+self.sub_batch_size]
                        avg_batch_len, avg_fwd_logprob, avg_bck_logprob = 0, 0, 0
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
                        if self.hps.get('write_mols_to_disk', False):
                            with open(self.gfn_samples_path+'/sampled_mols.pkl', 'wb') as f:
                                pickle.dump(mols, f)
                        rew_tup = self.reward.molecular_rewards(mols)
                        rew = torch.Tensor(rew_tup[2]).unsqueeze(dim=1)
                        if self.hps['task'] in self.task_model_reward_funcs:
                            normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, self.task_model, mols)
                        else:
                            normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, mols)
                        if self.hps['task_rewards_only']:
                            flat_rewards =  torch.Tensor(normalized_task_rew) #  #torch.mul(rew,flat_rewards_task) #flat_rewards_qed
                        else:
                            flat_rewards = rew*normalized_task_rew

                        #Drug likeliness score
                        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
                        if self.hps['diversity_filter']:
                            self.div_fil.update(smiles_list)
                            #penalize rewards for frequently generated scaffolds
                            flat_rewards = self.div_fil.penalize_reward(smiles_list,flat_rewards)

                        #Ensure reward is set to 0 for invalid trajectories (molecules)
                        pred_reward = torch.zeros((len(online_trajs), flat_rewards.shape[1]))
                        pred_reward[valid_idcs - 0] = flat_rewards.float()
                        
                        beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                        log_rewards = self.beta_to_logreward(beta_vector, pred_reward)

                        gfn_batch = self.algo.construct_batch(online_trajs, cond_info_encoding, log_rewards)#pred_reward.squeeze(dim=1))
                            
                        gfn_batch.num_online = len(online_trajs)
                        gfn_batch.num_offline = 0
                        gfn_batch.flat_rewards = pred_reward.detach().cpu()
                        gfn_batch.valid_percent =  len(valid_idcs)/len(online_trajs)
                        gfn_batch.unique_percent = len(set(smiles_list))/len(online_trajs)
                        gfn_batch.avg_batch_len = (avg_batch_len)/len(online_trajs)
                        gfn_batch.avg_fwd_logprob = avg_fwd_logprob/len(online_trajs)
                        gfn_batch.avg_bck_logprob = avg_bck_logprob/len(online_trajs)
                        
                        gfn_batch.avg_qed = np.average(rew_tup[1][3][0])
                        gfn_batch.avg_tpsa = np.average(rew_tup[1][0][0])
                        gfn_batch.avg_num_rings =  np.average(rew_tup[1][1][0])
                        gfn_batch.avg_sas =  np.average(rew_tup[1][2][0])
                        if (self.hps.get("objective",None)=='property_targeting'):
                            if (self.hps.subtype=='new_props'):
                                gfn_batch.avg_new_prop = np.average(rew_tup[1][4][0])
                        
                        gfn_batch.avg_zinc_rad = np.average(rew_tup[3])
                        gfn_batch.avg_task_reward = np.average(normalized_task_rew) #torch.mean(flat_rewards_task).item()
                        if self.hps.task is not None:
                            gfn_batch.avg_task_score = np.average(true_task_score)
                        else:
                            gfn_batch.avg_task_reward, gfn_batch.avg_task_score = 0,0
                                
                        yield gfn_batch, mols#, avg_batch_fwd_traj_len
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
    wrapped_model, dev = finetuner._wrap_for_mp(finetuner.gfn_trainer.model)
    if hasattr(finetuner.gfn_trainer, "model_prior") and finetuner.gfn_trainer.model_prior is not None:
        wrapped_model_prior,_ = finetuner._wrap_for_mp(finetuner.gfn_trainer.model_prior)
    else:
        wrapped_model_prior = None
    iterator = FTSampling_Iterator(finetuner,wrapped_model,wrapped_model_prior, dev, hps['sampling_batch_size'], hps['beta_exp'])
    train_loader = DataLoader(iterator,batch_size =None, num_workers = hps['num_workers'])
    return train_loader