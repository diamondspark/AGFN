# TODO: Give better class names
# from itertools import cycle
from typing import Iterator
from torch.utils.data import IterableDataset, DataLoader
from random import shuffle
from rdkit import Chem
import gzip
import torch
import pandas as pd
from torch.utils.data import Dataset

# class InMemoryDataset(Dataset):
#     def __init__(self, datafile, hps, batch_size, allow_charge=False):
#         self.df = pd.read_csv(hps['zinc_root']+datafile)
#         self.widx2df  = np.array_split(self.df, hps['num_workers'])
#         self.batch_size = batch_size
#         self.legal_atoms = set(["C", "N", "O", "F", "P", "S"])#set(pretrainer.ctx.atom_attr_values['v'])
#         self.allow_charge = allow_charge
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         worker_info = torch.utils.data.get_worker_info()
#         self._wid = worker_info.id if worker_info is not None else 0



class BatchSMILES(Dataset):
    'Given a list containing SMILES, returns batches of batch_size smiles'
    def __init__(self,smiles_list, batch_size):
        self.smiles_list = smiles_list
        self.batch_size = batch_size

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, index):
        start, end = index * self.batch_size, (index + 1) * self.batch_size
        batch_x = self.smiles_list[start:end]
        return batch_x, end, index+1



import numpy as np
class CustomDataset(Dataset):
  def __init__(self, filenames, batch_size, allow_charge=False):
        # batch_size = n_workers
    # `filenames` is a list of strings that contains all file names.
    # `batch_size` determines the number of files that we want to read in a chunk.
        self.filenames= filenames
        self.batch_size = batch_size
        self.legal_atoms = set(["C", "N", "O", "F", "P", "S"])#set(pretrainer.ctx.atom_attr_values['v'])
        self.allow_charge = allow_charge
  def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))   # Number of chunks.
  def __getitem__(self, idx): #idx means index of the chunk.
    # In this method, we do all the preprocessing.
    # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]   # This extracts one batch of file names from the list `filenames`.
        print('batch_x \n',batch_x)
        # data = []
        # labels = []
        # label_classes = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
        mols, valid_smiles = [],[]
        for file in batch_x:
            df = pd.read_csv(file)
            try:
                df['SMILES']
            except KeyError:
                df =  pd.read_csv(file, sep = '\t', header = None, names= ['SMILES', 'ZINCID'])
            for smiles in df['SMILES']:
                if self.allow_charge == False:
                    if ('+' in smiles) or ('-' in smiles):
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
            print(f'{file} finished len_mols {len(mols)}')
        if idx == self.__len__():  
          raise IndexError
        return valid_smiles
                

# custom_dataset = CustomDataset(Hfile_paths,batch_size=2)
# print('custom_dataset lenght ', custom_dataset.__len__())
# for i, smiles_list in enumerate(custom_dataset):
#     print(i, len(smiles_list))
#     smiles_batch = BatchSMILES(smiles_list,batch_size=5)
#     for j, batch in enumerate(smiles_batch):
#         print(j, len(batch))
#         if j ==5:
#             break
# # zinc_dataset = ZincNonIterableDataset(Hfile_paths)
# # zinc_loader = DataLoader(zinc_dataset, batch_size=4, num_workers=2)

# custom_loader = DataLoader(custom_dataset,batch_size=2, num_workers=2)
# for i, b in custom_loader:
#     print(i,b)


# n_workers = 2
# # custom_dataset = CustomDataset(Hfile_paths,batch_size=n_workers)
# per_worker_file_allotment = np.array_split(Hfile_paths,n_workers)

# def get_smiles_batch(f_idx, smiles_idx, wid, per_worker_file_allotment):
#     per_worker_files = per_worker_file_allotment[wid]
#     # custom_dataset = CustomDataset(per_worker_files,batch_size=1)
#     smiles_list = custom_dataset[f_idx] 
#     batch_smiles = BatchSMILES(smiles_list,batch_size=5)
#     smiles, end_idx =  batch_smiles[smiles_idx]
#     if end_idx>=len(smiles_list):
#         f_idx = max(len(per_worker_files),f_idx+1)
#     return smiles, end_idx, f_idx

#     for wid, per_worker_files in enumerate(per_worker_file_allotment):
#         print(f'worker {wid}, files {per_worker_file_allotment[wid]}')#files {per_worker_files}')
#         custom_dataset = CustomDataset(per_worker_file_allotment[wid],
#                                     batch_size=1)
#         smiles_batch = BatchSMILES(custom_dataset[j], batch_size=5)

# def get_smiles_batch(smiles_list, smiles_idx):
#     batch_smiles = BatchSMILES(smiles_list,batch_size=5)
#     smiles, end_idx =  batch_smiles[smiles_idx]
#     # if end_idx>=len(smiles_list):
#     #     f_idx = max(len(per_worker_files),f_idx+1)
#     return smiles, end_idx

# # print(per_worker_file_allotment)

# # # for i, smiles_list in enumerate(custom_dataset):
# # #     per_worker_file_allotment.append()
# # #     print(i,len())
# # f_idx, smiles_idx = 0,0
# # wid = 0
# # for i in range(10):
# #     per_worker_files = per_worker_file_allotment[wid]
# #     custom_dataset = CustomDataset(per_worker_files,batch_size=1)
# #     smiles_list = custom_dataset[f_idx]
# #     smiles, smiles_idx = get_smiles_batch(smiles_list, smiles_idx)
# #     print(smiles, smiles_idx)

# num_iter = 5000000000000
# class Sampling_Iterator(IterableDataset):
#     def __init__(self, pretrainer, mixer, Hfile_paths, num_workers, wrapped_model,dev,sub_batch_size, batch_size = 5) -> None:
#         super().__init__()
#         self.per_worker_file_allotment = np.array_split(Hfile_paths,num_workers)
#         self.f_idx, self.smiles_end_idx, self.smiles_idx = 0,0,0
#         self.batch_size = batch_size
#         self.model = wrapped_model
#         self.dev = dev
#         self.sub_batch_size = sub_batch_size
#         self.pretrainer = pretrainer
#         self.mixer = mixer

    
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         self._wid = worker_info.id if worker_info is not None else 0
#         per_worker_files = self.per_worker_file_allotment[self._wid]
#         for i in range(num_iter):
#             custom_dataset = CustomDataset(per_worker_files,batch_size=1)
#             smiles_list = custom_dataset[self.f_idx]
#             # shuffle(smiles_list)
#             batch_SMILES = BatchSMILES(smiles_list,batch_size=5)
#             while self.smiles_end_idx< len(smiles_list):
#                 # print('wid ', self._wid, 'self.smiles_idx ',self.smiles_idx,'smiles_endidx', self.smiles_end_idx)
#                 smiles_batch, self.smiles_end_idx, self.smiles_idx =  batch_SMILES[self.smiles_idx]

#                 mixed_traj_batch, avg_traj_len, offln_rew_tup = self.mixer.zinc_mix_trajectories(smiles_batch, self.model, self.dev)
#                 online_trajs, offline_trajs = mixed_traj_batch[0], mixed_traj_batch[1]
#                 assert (len(offline_trajs)+len(online_trajs)) % self.sub_batch_size == 0, "received an uneven batch size"
#                 r = (len(offline_trajs)+len(online_trajs)) // self.sub_batch_size

#                 for j in range(0,max(len(online_trajs),len(offline_trajs)),self.sub_batch_size):
#                     sub_online_trajs = online_trajs[j : j + self.sub_batch_size]
#                     valid_idcs = torch.tensor([i + 0 for i in range(len(sub_online_trajs)) if (sub_online_trajs[i + 0]["is_valid"]) & ("fwd_logprob" in sub_online_trajs[i+0])]).long()
#                     # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
#                     online_mols = [self.pretrainer.ctx.graph_to_mol(sub_online_trajs[i]['traj'][-1][0]) for i in valid_idcs]
#                     offline_mols = smiles_batch[j : j + self.sub_batch_size]

#                     online_rew_tup = self.pretrainer.reward.molecular_rewards(online_mols)
#                     online_rew = torch.Tensor(online_rew_tup[2]).unsqueeze(dim=1)
#                     #Ensure reward is set to 0 for invalid trajectories (molecules)
#                     pred_reward_online = torch.zeros(len(sub_online_trajs), online_rew.shape[1])
#                     pred_reward_online[valid_idcs - 0] = online_rew

#                     sub_offline_trajs = offline_trajs[j : j + self.sub_batch_size]
#                     offline_rew = offln_rew_tup[2][j : j + self.sub_batch_size]
#                     offline_rew = torch.Tensor(offline_rew).unsqueeze(dim=1)

#                     #combine online and offline rewards
#                     pred_reward = torch.cat((pred_reward_online,offline_rew))

#                     # calculate log(R(x)**beta) as log_rewards
#                     beta_vector = self.pretrainer.hps['beta']*torch.ones(pred_reward.shape[0])
#                     log_rewards = self.beta_to_logreward(beta_vector, pred_reward)

#                     #TODO: log_rewards is not needed now. pred_rewards are log_rewards now. 
#                     # Need to work on making gfn_batch. need cond_info encoding for this. This should be concatenate(cond_info online encoding, cond_info offline encoding)
#                     # print('samp_itr cond_info_online',self.mixer.traj_sampler.cond_info_encoding.shape, 
#                     # 'cond_info_bck', self.mixer.traj_sampler.cond_info_bck_encoding.shape)
#                     cond_info = torch.cat((self.mixer.traj_sampler.cond_info_encoding[j : j + self.sub_batch_size],
#                                         self.mixer.traj_sampler.cond_info_bck_encoding[j : j + self.sub_batch_size]))
                    
#                     # print('samp_itr_copy, cond_info', cond_info.shape, ' pred rewawrd ', pred_reward.shape)

#                     # gfn_batch = self.pretrainer.algo.construct_batch(sub_online_trajs+sub_offline_trajs, 
#                     #                                                     cond_info, pred_reward.squeeze(dim =1))
                    
#                     gfn_batch = self.pretrainer.algo.construct_batch(sub_online_trajs+sub_offline_trajs, 
#                                                                     cond_info, log_rewards)
                    
#                     if self.pretrainer.hps['validate_batch']:
#                         Utils.validate_batch( gfn_batch, mixed_traj_batch, self.pretrainer.ctx, self.pretrainer.gfn_trainer.model)
#                     gfn_batch.num_offline = self.mixer.num_offline // r
#                     gfn_batch.num_online = self.mixer.num_online // r
#                     gfn_batch.flat_rewards = pred_reward
#                     gfn_batch.online_flat_rewards = pred_reward_online
#                     gfn_batch.offline_flat_rewards = offline_rew
#                     # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
#                     gfn_batch.valid_percent = len(valid_idcs)/len(sub_online_trajs)
#                     # print('samp_iter.py total valid_idcs', len(valid_idcs), 'total online generated ', len(sub_online_trajs))
#                     # gfn_batch.preferences = self.pretrainer.cond_info.get("preferences", None)
#                     # gfn_batch.focus_dir = self.pretrainer.cond_info.get("focus_dir", None)
#                     # gfn_batch.smiles = smiles_batch
#                     # print('samp itr smiles_batch ',smiles_batch)
#                     # with open('./test_log.txt','a+') as f:
#                     #     f.write(str(self._wid) + str(smiles_batch)+'\n')
#                     yield gfn_batch, avg_traj_len, online_mols, offline_mols
                
#             self.f_idx+=1
#             self.smiles_end_idx, self.smiles_idx = 0,0
        

#     def beta_to_logreward(self,beta_vector, pred_reward):
#         scalar_logreward = pred_reward.squeeze().clamp(min=1e-30).log()
#         assert len(scalar_logreward.shape) == len(
#             beta_vector.shape
#         ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {beta_vector.shape}"
#         return scalar_logreward * beta_vector   # log(r(x)**beta) = beta*log(r(x))
     
# iterator = Sampling_Iterator(Hfile_paths, num_workers=2)
# train_loader = DataLoader(iterator,batch_size =None, num_workers = 2)
# # ctr = 0
# # while True:
# for i, b in enumerate(train_loader):
#     print(i,b)
#     # if i ==10: #hps['num_iter]
#     #     break