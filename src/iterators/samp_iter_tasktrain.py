import torch
from torch.utils.data import DataLoader, IterableDataset
from rdkit import Chem
from functools import partial
from agfn.backtraj import ReverseFineTune
import numpy as np

class TaskSampling_Iterator(IterableDataset):
    def __init__(self, task_trainer, wrapped_model, dev, num_online, beta, task_conditionals_dict=None ):
        self.num_online = num_online
        self.cond_info_task = task_trainer.cond_info_task
        self.wrapped_model = wrapped_model
        self.dev = dev
        self.beta = beta
        self.hps = task_trainer.hps
        self.graph_sampler = task_trainer.graph_sampler
        self.task_conditionals_dict = task_conditionals_dict
        # self.task = ft_trainer.task
        self.ctx = task_trainer.ctx
        self.algo = task_trainer.algo
        self.reward = task_trainer.reward
        self.task_model = task_trainer.task_model
        self.reverse = ReverseFineTune(task_trainer.env, self.ctx,self.hps, task_trainer.rng)
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
            self.Y_scaler = task_trainer.Y_scaler
        self.sub_batch_size = task_trainer.hps['training_batch_size']
        # self.task_model_reward_funcs = {
        #     'Caco2': partial(self.reward.caco2, self.Y_scaler, self.task_model),
        #     'LD50': partial(self.reward.ld50, self.Y_scaler, self.task_model),
        #     'Lipophilicity': partial(self.reward.lipophilicity, self.Y_scaler, self.task_model),
        #     'Solubility': partial(self.reward.solubility, self.Y_scaler, self.task_model),
        #     'BindingRate': partial(self.reward.binding_rate, self.Y_scaler, self.task_model),
        #     'MicroClearance': partial(self.reward.micro_clearance, self.Y_scaler, self.task_model),
        #     'HepatocyteClearance': partial(self.reward.hepatocyte_clearance, self.Y_scaler, self.task_model)
        # }
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
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        for _ in range(self.hps['num_iter']):
            try:
                cond_info = self.cond_info_task.compute_cond_info_forward(self.num_online)
                cond_info_encoding = self.cond_info_task.thermometer_encoding(cond_info)
                if self.task_conditionals_dict:
                    cond_info_tasktrain = self.cond_info_task.compute_cond_info_forward(self.num_online, self.task_conditionals_dict)
                    cond_info_tasktrain_encoding = self.cond_info_task.thermometer_encoding(cond_info_tasktrain, self.task_conditionals_dict).to(self.dev)
                else:
                    cond_info_tasktrain_encoding = None

                with torch.no_grad():
                    online_trajs_sampled = self.graph_sampler.sample_from_model(self.wrapped_model,self.num_online,
                                                                            cond_info_encoding.to(self.dev),
                                                                            self.dev, random_stop_action_prob= self.hps['random_stop_prob'],
                                                                            random_action_prob = self.hps['random_action_prob'],
                                                                            ft_cond_info = cond_info_tasktrain_encoding)
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
                    rew = torch.Tensor(rew_tup[2]).unsqueeze(dim=1)
                    if self.hps['task'] in self.task_model_reward_funcs:
                        normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, self.task_model, mols)
                    else:
                        normalized_task_rew, true_task_score = self.reward.task_reward(self.hps.task, self.task_model, mols)

                    if self.hps['task_rewards_only']:
                        flat_rewards =  torch.Tensor(normalized_task_rew) #  #torch.mul(rew,flat_rewards_task) #flat_rewards_qed
                    else:
                        flat_rewards = rew*normalized_task_rew

                    #Drug likeliness score
                    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]

                    #Ensure reward is set to 0 for invalid trajectories (molecules)
                    pred_reward = torch.zeros((len(online_trajs), flat_rewards.shape[1]))
                    pred_reward[valid_idcs - 0] = flat_rewards.float()

                
                    beta_vector = self.hps['beta_exp']*torch.ones(pred_reward.shape[0])
                    log_rewards = self.beta_to_logreward(beta_vector, pred_reward)
                    if self.hps.task_conditionals:
                        gfn_batch = self.algo.construct_batch(online_trajs, cond_info_encoding,cond_info_tasktrain_encoding, log_rewards)#pred_reward.squeeze(dim=1))
                    else:
                        gfn_batch = self.algo.construct_batch(online_trajs, cond_info_encoding, log_rewards)#pred_reward.squeeze(dim=1))
                    gfn_batch.num_online = len(online_trajs)
                    gfn_batch.num_offline = 0
                    gfn_batch.flat_rewards = pred_reward.detach().cpu()
                    gfn_batch.valid_percent =  len(valid_idcs)/len(online_trajs)
                    gfn_batch.unique_percent = len(set(smiles_list))/len(online_trajs)
                    gfn_batch.avg_batch_len = (avg_batch_len)/len(online_trajs)
                    # gfn_batch.sa_score = np.average([sascore.calculateScore(m) for m in mols])
                    gfn_batch.avg_fwd_logprob = avg_fwd_logprob/len(online_trajs)
                    gfn_batch.avg_bck_logprob = avg_bck_logprob/len(online_trajs)
                    # gfn_batch.avg_mol_wt = np.average(rew_tup[1][0][0])
                    # gfn_batch.avg_logP = np.average(rew_tup[1][1][0])
                    gfn_batch.avg_qed = np.average(rew_tup[1][3][0])
                    gfn_batch.avg_tpsa = np.average(rew_tup[1][0][0])
                    gfn_batch.avg_num_rings =  np.average(rew_tup[1][1][0])
                    gfn_batch.avg_sas =  np.average(rew_tup[1][2][0])
                    if (self.hps.objective=='property_targeting'):
                        if(self.hps.subtype=='new_props'):
                            gfn_batch.avg_new_prop = np.average(rew_tup[1][4][0])
                    
                    gfn_batch.avg_zinc_rad = np.average(rew_tup[3])
                    gfn_batch.avg_task_reward = np.average(normalized_task_rew) #torch.mean(flat_rewards_task).item()
                    if self.hps.task is not None:
                        gfn_batch.avg_task_score = np.average(true_task_score)
                    else:
                        gfn_batch.avg_task_reward, gfn_batch.avg_task_score = 0,0
                            
                    yield gfn_batch, mols#, avg_batch_fwd_traj_len
            except Exception as e:
                print(e, len(mols))
                continue
    def beta_to_logreward(self,beta_vector, pred_reward):
        scalar_logreward = pred_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            beta_vector.shape
        ), f"dangerous shape misatch: {scalar_logreward.shape} vs {beta_vector.shape}"
        return scalar_logreward * beta_vector   # log(r(x)**beta) = beta*log(r(x))

            
def build_train_loader(hps,tasktrainer,task_conditionals_dict):
    wrapped_model, dev = tasktrainer._wrap_for_mp(tasktrainer.gfn_trainer.model)
    iterator = TaskSampling_Iterator(tasktrainer,wrapped_model,dev, hps['sampling_batch_size'], hps['beta_exp'], 
                                     task_conditionals_dict=task_conditionals_dict)
    train_loader = DataLoader(iterator,batch_size =None, num_workers = hps['num_workers'])
    return train_loader