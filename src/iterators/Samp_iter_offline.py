import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader, IterableDataset

class Sampling_Iterator_Offline(IterableDataset):
    def __init__(self, datafile, pretrainer, wrapped_model, dev, mixer) -> None:
        num_workers = pretrainer.hps['num_workers']
        self.smiles_idx = 0
        self.batch_size = pretrainer.hps['sampling_batch_size']
        self.model = wrapped_model
        self.dev = dev
        self.sub_batch_size = pretrainer.hps['training_batch_size']
        # self.pretrainer = pretrainer
        self.mixer = mixer
        self.algo = pretrainer.algo
        self.ctx = pretrainer.ctx
        self.task = pretrainer.task
        self.reward = pretrainer.reward
        self.hps = pretrainer.hps
        self.legal_atoms = set(pretrainer.hps['atoms'])#set(["C", "N", "O", "F", "P", "S"])#set(pretrainer.ctx.atom_attr_values['v'])
        self.allow_charge = True if len(pretrainer.ctx.charges)>1 else False
        self.df = pd.read_csv(self.hps['zinc_root']+datafile)#.head(1000)
        if self.hps['preprocess_offln_data']:
            self.valid_smiles = self.preprocess(self.df)
        else:
            self.valid_smiles = self.df.smiles
        self.widx2smiles = np.array_split(self.valid_smiles, self.hps['num_workers'])
        print('samp_iteroffln len val smiles ',len(self.valid_smiles), self.valid_smiles[0:5])
        
    def preprocess(self, df):
        mols, valid_smiles = [], []
        for smiles in df['smiles']:
            if self.allow_charge == False:
                if ('+' in smiles) or ('-' in smiles) or ('.' in smiles):  #TODO better logic for removing charged mols. CC1(C)CC(=O)C2=C(C1)Nc1[nH]c(=O)[nH]c(=O)c1[C@H]2c1ccc(-c2cc(Cl)ccc2Cl)o1 will fail with current logic
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
      
    def beta_to_logreward(self,beta_vector, pred_reward):
        scalar_logreward = pred_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            beta_vector.shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {beta_vector.shape}"
        return scalar_logreward * beta_vector   # log(r(x)**beta) = beta*log(r(x))  

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        worker_smiles = self.widx2smiles[self._wid]
        for i in range(self.hps['num_iter']):
            self.smiles_idx = 0
            while self.smiles_idx < len(worker_smiles):
                try:
                    smiles_batch = worker_smiles[self.smiles_idx:self.smiles_idx+self.batch_size//2]  # TODO: Account for some smiles left out at the end of the worker_smiles list
                    self.smiles_idx += self.batch_size//2  # divided by 2 because smiles_batch is batch_size/2; the remainder batch_size/2 is for online trajs
                    # print('sampiteroffln.py ready to sample trajs, self.dev', self.dev, 'smiles batch off ',smiles_batch)
                    with torch.no_grad():
                        mixed_traj_batch, avg_traj_len, offln_rew_tup = self.mixer.zinc_mix_trajectories(smiles_batch, self.model, self.dev)
                        # print('sampiteroff.py offline consolidated reward ', offln_rew_tup[0])
                    online_trajs, offline_trajs = mixed_traj_batch[0], mixed_traj_batch[1]
                    # assert (len(offline_trajs)+len(online_trajs)) % self.sub_batch_size == 0, "received an uneven batch size"
                    r = (len(offline_trajs)+len(online_trajs)) // self.sub_batch_size
                    # print('sampiterinmmewm offline trajs ' , len(offline_trajs), 'online trajs ', len(online_trajs)) 
                    for j in range(0,max(len(online_trajs),len(offline_trajs)),self.sub_batch_size):
                        sub_online_trajs = online_trajs[j : j + self.sub_batch_size]
                        # print('j', j, 'sub_online_trajs ', len(sub_online_trajs))
                        valid_idcs = torch.tensor([i + 0 for i in range(len(sub_online_trajs)) if (sub_online_trajs[i + 0]["is_valid"]) & ("fwd_logprob" in sub_online_trajs[i+0])]).long()
                        # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
                        online_mols = [self.ctx.graph_to_mol(sub_online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
                        smiles_list = [Chem.MolToSmiles(mol) for mol in online_mols]
                        offline_mols = smiles_batch[j : j + self.sub_batch_size]
                        online_rew_tup = self.reward.molecular_rewards(online_mols)
                        # print('samp iter online reward tup',online_rew_tup[0], online_rew_tup[1],online_rew_tup[2] )
                        online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
                        #Ensure reward is set to 0 for invalid trajectories (molecules)
                        pred_reward_online = torch.zeros(len(sub_online_trajs), online_rew.shape[1])
                        pred_reward_online[valid_idcs - 0] = online_rew

                        sub_offline_trajs = offline_trajs[j : j + self.sub_batch_size]
                        offline_rew = offln_rew_tup[2][j : j + self.sub_batch_size]
                        offline_rew = torch.Tensor(offline_rew).unsqueeze(dim=1)
                        # print('samp iter offline_rew ',offline_rew, offln_rew_tup[1][j : j + self.sub_batch_size] )
                        #offline rewards only
                        pred_reward = offline_rew #torch.cat((pred_reward_online,offline_rew))
                        # print('samp iter pred reward ', pred_reward)

                        # calculate log(R(x)**beta) as log_rewards
                        beta_vector = self.hps['beta']*torch.ones(pred_reward.shape[0])
                        log_rewards = self.beta_to_logreward(beta_vector, pred_reward)

                        #TODO: log_rewards is not needed now. pred_rewards are log_rewards now. 
                        # Need to work on making gfn_batch. need cond_info encoding for this. This should be concatenate(cond_info online encoding, cond_info offline encoding)
                        # print('samp_itr cond_info_online',self.mixer.traj_sampler.cond_info_encoding.shape, 
                        # 'cond_info_bck', self.mixer.traj_sampler.cond_info_bck_encoding.shape)
                        # print(f'Samp iter.py wid {self._wid}, start (j) {j}, end (j+subbatch) {j + self.sub_batch_size}')
                        cond_info = self.mixer.traj_sampler.cond_info_bck_encoding[j : j + self.sub_batch_size]
                        # torch.cat((self.mixer.traj_sampler.cond_info_encoding[j : j + self.sub_batch_size],
                                            # self.mixer.traj_sampler.cond_info_bck_encoding[j : j + self.sub_batch_size]))
                        
                        # print('samp_itr_copy, cond_info', cond_info.shape, ' pred rewawrd ', pred_reward.shape)

                        # gfn_batch = self.pretrainer.algo.construct_batch(sub_online_trajs+sub_offline_trajs, 
                        #                                                     cond_info, pred_reward.squeeze(dim =1))
                        gfn_batch = self.algo.construct_batch(sub_offline_trajs, #sub_online_trajs+sub_offline_trajs, 
                                                                        cond_info, log_rewards)
                        # print('sampiterinmem.py gfn_batch created')
                                                
                        gfn_batch.num_sub_online_trajs = 0# len(sub_online_trajs)
                        gfn_batch.num_sub_offline_trajs = len(sub_offline_trajs)
                        gfn_batch.num_offline = self.mixer.num_offline // r
                        gfn_batch.num_online = self.mixer.num_online // r
                        gfn_batch.flat_rewards = pred_reward
                        gfn_batch.online_flat_rewards = pred_reward_online
                        gfn_batch.offline_flat_rewards = offline_rew
                        # print('sampiterinmem.py avg offline rew ',  torch.mean(gfn_batch.offline_flat_rewards).item())
                        # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
                        gfn_batch.valid_percent = len(valid_idcs)/len(sub_online_trajs)
                        gfn_batch.unique_percent = len(set(smiles_list))/len(sub_online_trajs)
                        gfn_batch.offln_zinc_rad = np.average(offln_rew_tup[3][j : j + self.sub_batch_size])
                        gfn_batch.onln_zinc_rad = np.average(online_rew_tup[3])
                       
                        # gfn_batch.offline_avg_mol_wt = np.average(offln_rew_tup[1][0][0][j:j+self.sub_batch_size])
                        # gfn_batch.online_avg_mol_wt = np.average(online_rew_tup[1][0][0])
                        # gfn_batch.offline_avg_fsp3 = np.average(offln_rew_tup[1][1][0][j:j+self.sub_batch_size])
                        # gfn_batch.online_avg_fsp3 = np.average(online_rew_tup[1][1][0])
                        # gfn_batch.offline_avg_logP = np.average(offln_rew_tup[1][1][0][j:j+self.sub_batch_size])
                        # gfn_batch.online_avg_logP =np.average(online_rew_tup[1][1][0])
                        # gfn_batch.offline_avg_num_rot_bonds = np.average(offln_rew_tup[1][3][0][j:j+self.sub_batch_size])
                        # gfn_batch.online_avg_num_rot_bonds = np.average(online_rew_tup[1][3][0])
                        gfn_batch.offline_avg_tpsa =  np.average(offln_rew_tup[1][0][0][j:j+self.sub_batch_size])
                        gfn_batch.online_avg_tpsa =  np.average(online_rew_tup[1][0][0])
                        gfn_batch.offln_num_rings = np.average(offln_rew_tup[1][1][0][j:j + self.sub_batch_size])
                        gfn_batch.online_num_rings = np.average(online_rew_tup[1][1][0])


                        gfn_batch.offline_avg_SAS = np.average(offln_rew_tup[1][2][0][j:j+self.sub_batch_size])
                        gfn_batch.online_avg_SAS =  np.average(online_rew_tup[1][2][0])
                        gfn_batch.offline_qed = np.average(offln_rew_tup[1][-1][0][j:j+self.sub_batch_size])
                        gfn_batch.online_qed = np.average(online_rew_tup[1][-1][0])
                        # print('sampiter.py offline_qed', gfn_batch.offline_qed)
                        # print('sampiter.py online_qed', gfn_batch.online_qed)

                        # print('sampiter.py offline_rew_zinc_rad', gfn_batch.offln_zinc_rad)#5][0][j:j + self.sub_batch_size], 'j', j, 'self.sub_batch_size ',self.sub_batch_size)#offln_rew_tup[1][5][0] ) #  offln_rew_tup[1][5][j:j + self.sub_batch_size][0]
                        # print('sampiter.py onln_rew_zinc_rad', gfn_batch.onln_zinc_rad)#[5][0])#online_rew_tup[1][5][0])  # online_rew_tup[1][5][0]

                        # print(f'samp iter zinc flat rad offln {gfn_batch.offln_zinc_rad} onln zinc flat {gfn_batch.onln_zinc_rad}, {np.average(gfn_batch.onln_zinc_rad)}')
                        # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
                        # gfn_batch.preferences = self.pretrainer.cond_info.get("preferences", None)
                        # gfn_batch.focus_dir = self.pretrainer.cond_info.get("focus_dir", None)
                        # gfn_batch.smiles = smiles_batch
                        # print('samp itr smiles_batch ',smiles_batch)
                        # with open('./test_log.txt','a+') as f:
                        #     f.write(str(self._wid) + str(smiles_batch)+'\n')
                        yield gfn_batch, avg_traj_len, None, online_mols, offline_mols
                except Exception as e:
                    print(e)
                    print(f'wid {self._wid}, self.smiles_idx , {self.smiles_idx}')
                    # print(f' smiles {[smiles_batch[j : j + self.sub_batch_size][i] for i in valid_idcs]}')
                    # if str(e)!='3 is not in list':
                    #     raise e
                    # raise e
                    continue 

def build_train_loader(hps, pretrainer, mixer, datafile=None):
    # num_online, num_offline, cond_info = pretrainer.mixing_counts(hps['gfn_batch_size'],hps['mix_ratio'])
    # mixer = TrajectoryMixer(pretrainer,num_online, num_offline,replay=hps['replay'])
    # Hfile_paths = Utils.get_Hfile_paths(hps['zinc_root'])
    # Hfile_paths = ['/mnt/ps/home/CORP/mohit.pandey/project/files.docking.org/zinc22/2d-all/H04/H04P100.smi.gz',
    #             '/mnt/ps/home/CORP/mohit.pandey/project/files.docking.org/zinc22/2d-all/H04/H04M100.smi.gz']
    wrapped_model, dev = pretrainer._wrap_for_mp(pretrainer.gfn_trainer.model, send_to_device=True)
    mixer.traj_sampler.model = wrapped_model
    # print('samp iter wrapped_model , dev', wrapped_model, dev)
    iterator = Sampling_Iterator_Offline(datafile,pretrainer, wrapped_model,dev, mixer)
    # for b in iterator:
    #     print('samp iter ', b)
    #     break
    train_loader = DataLoader(iterator,batch_size =None, num_workers = hps['num_workers'])
    return train_loader

