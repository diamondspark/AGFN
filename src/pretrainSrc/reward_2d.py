from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem import Descriptors, Crippen, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcFractionCSP3
from rdkit import Chem
import torch
import pickle
import numpy as np
from torch.nn.functional import softplus
from gflownet.utils import sascore
from rdkit.Chem import QED
from typing import List
from utils.maplight import *
import pandas as pd
# from Energy_Utils.energy_utils import create_pytorch_geometric_graph_data_list_from_smiles_and_labels

def scale_range( OldValue, OldMax, OldMin, NewMax, NewMin):
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def calculate_tanimoto_similarity(mol1, mol2):
        """
        Calculate the Tanimoto similarity between two molecules given their SMILES strings.
        Args:
        smiles1 (str): The SMILES string of the first molecule.
        smiles2 (str): The SMILES string of the second molecule.
        Returns:
        float: The Tanimoto similarity between the two molecular fingerprints.
        """
        # # Convert SMILES to molecule objects
        # mol1 = Chem.MolFromSmiles(smiles1)
        # mol2 = Chem.MolFromSmiles(smiles2)
        # # Check for successful molecule creation
        # if mol1 is None or mol2 is None:
        #     raise ValueError("Invalid SMILES string provided. Check your SMILES strings.")
        # Generate fingerprints
        fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator()
        fp1 = fp_gen.GetFingerprint(mol1)
        fp2 = fp_gen.GetFingerprint(mol2)
        # Calculate Tanimoto similarity
        tanimoto_sim = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto_sim

class Reward():
    def __init__(self, conditional_range_dict, cond_prop_var,reward_aggregation, molenv_dict_path, zinc_rad_scale, hps, wrapped_glidecnn_model=None) -> None:
        self.cond_range = conditional_range_dict
        self.cond_var = cond_prop_var
        self.reward_aggregation = reward_aggregation
        self.zinc_rad_scale = zinc_rad_scale
        with open(molenv_dict_path, "rb") as f:
            self.atomenv_dictionary = pickle.load(f)
        self.glidecnn_model = wrapped_glidecnn_model
        self.hps = hps
        if 'seed_smiles' in self.hps:
            self.seedmol = Chem.MolFromSmiles(hps['seed_smiles'])
        else:
            self.seedmol = None

        # self.precompute_normalizing_const()

    def mol_wt(self, mols: List[RDMol]):
        flat_mw_reward = np.array([Descriptors.MolWt(mol) for mol in mols])#.unsqueeze(dim=-1)
        return flat_mw_reward
    
    def qed(self, mols: List[RDMol]):
        flat_qed_reward = np.array([QED.qed(mol) for mol in mols])
        return flat_qed_reward
    
    def logP(self, mols: List[RDMol]):
        # RDKit uses Crippen under the hood for this which is a reliable albeit a predictive model.
        # Do we want to use it? 
        flat_logP_reward = np.array([Crippen.MolLogP(mol) for mol in mols])#.unsqueeze(dim=-1)
        return flat_logP_reward
    
    def tpsa(self, mols: List[RDMol]):
        flat_mw_reward = np.array([CalcTPSA(mol) for mol in mols])#.unsqueeze(dim=-1)
        return flat_mw_reward
    
    def fsp3(self, mols: List[RDMol]):
        flat_mw_reward = np.array([CalcFractionCSP3(mol) for mol in mols])#.unsqueeze(dim=-1)
        return flat_mw_reward
    
    def count_rotatable_bonds(self, mols: List[RDMol]):
        flat_mw_reward = np.array([CalcNumRotatableBonds(mol) for mol in mols])#.unsqueeze(dim=-1)
        return flat_mw_reward
    
    def count_num_rings(self, mols: List[RDMol]):
        def count_5_and_6_membered_rings(mol):
            ring_info = mol.GetRingInfo()
            num_five_six_membered_rings = len([ring for ring in ring_info.AtomRings() if (len(ring) == 6) or (len(ring) == 5) ])
            # num_five_membered_rings = len([ring for ring in ring_info.AtomRings() if len(ring) == 5])
            return num_five_six_membered_rings # num_five_membered_rings+num_six_membered_rings

        # flat_ring_reward = np.array([mol.GetRingInfo().NumRings() for mol in mols ])
        flat_ring_reward = np.array([count_5_and_6_membered_rings(mol) for mol in mols])
        return flat_ring_reward
    
    def synthetic_assessibility(self,  mols: List[RDMol]):
        sas_flat_reward = np.array([sascore.calculateScore(mol) for mol in mols])
        return sas_flat_reward

    def permeability(self, caco2_model, mols: List[RDMol] ):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(caco2_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        caco2_list = np.array([caco2_model.predict(x) for x in mol_fing])
        return caco2_list
    
    def toxicity(self, tox_model, mols: List[RDMol] ):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        # ld50_list = np.array([tox_model.predict(x) for x in mol_fing])
        y_pred1 = self.Y_scaler.inverse_transform(tox_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        # print('reward2d.py tox ', y_pred1)

        # ypred1_normal_fn = lambda x: scale_range(x, 6, 0, 1, 0)
        # y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
        return y_pred1 #ld50_list

    def lipo(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        lipo_list = np.array([task_model.predict(x) for x in mol_fing])
        return lipo_list

    def sol(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        sol_list = np.array([task_model.predict(x) for x in mol_fing])
        return sol_list

    def bind(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        bind_list = np.array([task_model.predict(x) for x in mol_fing])
        return bind_list

    def mclear(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        mclear_list = np.array([task_model.predict(x) for x in mol_fing])
        return mclear_list

    def hclear(self, task_model, mols):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        mol_fing = get_fingerprints(pd.Series(smiles_list))
        y_pred1 = self.Y_scaler.inverse_transform(task_model.predict(mol_fing, thread_count=32)).reshape(-1,1)
        return y_pred1
        hclear_list = np.array([task_model.predict(x) for x in mol_fing])
        return hclear_list


    def docking_reward(self, mols, seed_mol):
        if self.glidecnn_model:
            device = next(self.glidecnn_model.parameters()).device
            # print('reward2d.py device ', device)
            # smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
            docking_scores = glide_cnn_scores(self.glidecnn_model, device, mols, batch_size= 64) #len(mols))
            docking_scores_normalized = -docking_scores/10
        else:
            docking_scores = docking_scores_normalized = np.array([1]*len(mols))
        # sim = np.array([min(0.4, calculate_tanimoto_similarity(mol, self.seedmol))/0.4 for mol in mols])
        sim = np.array([1]*len(mols))

        similarity_values = []
        for mol in mols:
            # Calculate Tanimoto similarity
            similarity = calculate_tanimoto_similarity(mol, self.seedmol)
            
            if similarity >= 0.9:
                similarity_values.append(0)  # Assign 0 if similarity >= 0.9
            else:
                # Linearly increasing value from 0 to 1 based on similarity (for similarity < 0.9)
                value = 1 - similarity / 0.9

                similarity_values.append(value)
        
        # Convert the list to a numpy array
        sim = np.array(similarity_values)


        docking_scores_normalized = np.ones_like(docking_scores_normalized) # Uncomment when docking score is needed.
        docking_flatreward = docking_scores_normalized*sim
        # print('reward_2d.py dockingflat reward' , docking_flatreward.shape, docking_flatreward)
        # if self.seedmol:
        #     sim = np.array([min(0.4, calculate_tanimoto_similarity(mol, self.seedmol))/0.4 for mol in mols]) #only for logging 

        return docking_flatreward, docking_scores, sim  
    
    def unidocking_reward(self, mols, seed_mol):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        # docking_flatreward=np.array([1]*len(mols))
        docking_scores = np.array(unidock_scores(smiles_list)).astype(np.float64)
        docking_scores_normalized = -docking_scores/10
        sim = np.array([min(0.4, calculate_tanimoto_similarity(mol, seed_mol))/0.4 for mol in mols])
        docking_flatreward = docking_scores_normalized*sim
        # print('reward_2d.py dockingflat reward' , docking_flatreward.shape, docking_flatreward)
        return docking_flatreward, docking_scores, sim


    def energy_reward( energy_classfn_model, energy_regressn_model, mols: List[RDMol], batch_size=None):
        data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(mols,[0]*len(mols))
        classn_device = next(energy_classfn_model.parameters()).device
        regressn_device = next(energy_regressn_model.parameters()).device
         #TODOD: torch.no_grad; expt. with batchsize here; MLP with morgan for energy
        def energy_filter(energy_classfn_model, data_list):
            dataloader = DataLoader(data_list, batch_size = len(mols) if not batch_size else batch_size)
            for batch in dataloader: 
                with torch.no_grad():
                    output = model_classn(batch.to(classn_device))
                predicted_proba = torch.sigmoid(output.detach().cpu())
                thresh = Variable(torch.Tensor([0.5]))  # threshold
                y_pred_tag = (predicted_proba > thresh).squeeze().float().detach().cpu().numpy()
                valid_idcs = np.where(y_pred_tag == 1.0)[0]
                valid_mols = list(np.array(smiles_list)[valid_idcs])
                
            return valid_mols, valid_idcs

        def energy_score(energy_regressn_model, valid_mols, valid_idcs, batch_size=None):
            valid_datalist = [data_list[i] for i in valid_idcs]
            dataloader = DataLoader(valid_datalist, batch_size = len(mols))
            node_num_list = [data.x.shape[0] for data in valid_datalist]

            for batch in dataloader:
                with torch.no_grad():
                    normalized_energy = model_normalized_regress(batch.to(device)).detach().cpu().numpy().squeeze()
                energy = np.array(node_num_list)*np.exp(normalized_energy)

            return normalized_energy, energy

        energy_flatreward = np.zeros((len(mols)))
        valid_mols, valid_idcs = energy_filter(energy_classfn_model, data_list)
        normalized_energy, energy = energy_score(energy_regressn_model, valid_mols, valid_idcs)
        energy_flatreward[valid_idcs] = normalized_energy
        # energy_reward[valid_idcs] = normalized_energy
        unnormalized_energy = np.zeros((len(mols)))
        unnormalized_energy[valid_idcs] = energy

        return energy_flatreward, unnormalized_energy


    def searchAtomEnvironments_fraction(self,  mols: List[RDMol], radius=2):
        def per_mol_fraction(mol, radius):
            info = {}
            atomenvs = 0
            AllChem.GetMorganFingerprint(
                mol,
                radius,
                bitInfo=info,
                includeRedundantEnvironments=True,
                useFeatures=False,
            )
            for k, v in info.items():
                for e in v:
                    if e[1] == radius:
                        if k in self.atomenv_dictionary:
                            atomenvs += 1
            return atomenvs / max(mol.GetNumAtoms(), 1)
        return [per_mol_fraction(mol,radius)+np.finfo(np.float32).eps+1e-8 for mol in mols] # add \epsilon to prevent 0 composite reward upon multiplication of all flat rewards

    

    # def caco2(self,Y_scaler,task_model,  mols: List[RDMol]):
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))  #from maplight
    #     y_pred1 =  Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, -3, -8, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized
    #     # flat_caco_reward = np.array([np.exp(-s) for s in y_pred1])
    #     flat_caco_reward = np.array([1*1/(1+(np.exp(0.2*(s+10)))) for s in y_pred1])
    #     # print('reward 2d.py flat_caco)Reward ', flat_caco_reward.shape)
    #     return flat_caco_reward # FlatRewards(torch.as_tensor(flat_caco_reward))

    # def ld50(self,Y_scaler,task_model,  mols: List[RDMol]):
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=32)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 6, 0, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized

    # def lipophilicity(self, Y_scaler, task_model, mols):
    #     # Implement lipophilicity prediction using task_model
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 5, -2, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized
    #     pass

    # def solubility(self, Y_scaler, task_model, mols):
    #     # Implement solubility prediction using task_model
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 2, -13, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized

    # def binding_rate(self, Y_scaler, task_model, mols):
    #     # Implement binding rate prediction using task_model
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 100, 0, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized
        

    # def micro_clearance(self, Y_scaler, task_model, mols):
    #     # Implement micro-clearance prediction using task_model
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 200, 0, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized

    # def hepatocyte_clearance(self, Y_scaler, task_model, mols):
    #     # Implement hepatocyte clearance prediction using task_model
    #     smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    #     X_test1 = get_fingerprints(pd.Series(smiles_list))
    #     y_pred1 = Y_scaler.inverse_transform(task_model.predict(X_test1, thread_count=1)).reshape(-1,1)
    #     ypred1_normal_fn = lambda x: scale_range(x, 200, 0, 1, 0)
    #     y_pred1_normalized = ypred1_normal_fn(np.array(y_pred1))
    #     return y_pred1_normalized


    def _compositional_reward(self, flat_reward, property, range, slope, rate = 1):
        '''
            flat_reward: list of 1 property for each mol in the batch
            slope : -1 when lower is better (y=-x), +1 otherwise (y=x)
            TODO: DECAY exp at faster rate when p_x<p_0 and slope =1; similarly,
            for p_x>p1 & slope = -1.
        '''
        if (property == 'zinc_radius') : #or (property == 'qed'):
            return flat_reward
        else:
            composite_reward = []
            for p_x in flat_reward:
                try:
                    range[0]
                except Exception as e:
                    print('reward_2d, px, range0, range1, slope', p_x, range, slope, property)
                if p_x<range[0]:
                    if slope == 1:
                        reward = normalized_reward =  0.5*np.exp(-(range[0]-p_x)/rate)
                    elif slope == -1:
                        reward = normalized_reward = np.exp(-(range[0]-p_x)/rate)
                    elif slope ==0:
                        normalized_reward = np.exp(-(range[0]-p_x)/rate)
                elif p_x>range[1]:
                    if slope == 1:
                        reward = normalized_reward = np.exp(-(p_x-range[1])/rate)
                    elif slope == -1:
                        reward = normalized_reward = 0.5*np.exp(-(p_x-range[1])/rate)
                    elif slope == 0:
                        normalized_reward = np.exp(-(p_x-range[1])/rate)
                else:
                    if slope ==1:
                        normalized_reward = reward = 0.5*((p_x-range[0])/(range[1]-range[0])) + 0.5
                        # normalized_reward = (p_x-range[0])/(range[1]-range[0])
                    elif slope ==-1:
                        normalized_reward = reward = -0.5*((p_x-range[0])/(range[1]-range[0])) + 1
                        # normalized_reward = 1 - ((p_x-range[0])/(range[1]-range[0]))
                    elif slope ==0:
                        normalized_reward = reward = 1  
                composite_reward.append(normalized_reward)
        
        return composite_reward


    def _consolidate_rewards(self,flat_rewards):
        lg_rewards = []
        for (flat_reward, property, range, slope) in flat_rewards:
            try:
                rate = self.cond_var[property]
            except KeyError as e:
                rate = 1 # Used when conditioning on new properties during finetuning. We dont define a cond_var for new props.
            lg_rewards.append(self._compositional_reward(flat_reward,property,range,slope, rate))
        return np.array(lg_rewards).T   # transposing makes sure that we return (n_mols, n_props)

    def _compute_flat_rewards(self, mols):
        flat_rewards = []
        for property in self.cond_range.keys():
            # print('reward2d.py property being considered ', property)
            if property == 'Mol_Wt':
                flat_reward = self.mol_wt(mols)
            elif property == 'fsp3':
                flat_reward = self.fsp3(mols)
            elif property == 'logP':
                flat_reward = self.logP(mols)
            elif property == 'num_rot_bonds':
                flat_reward = self.count_rotatable_bonds(mols)
            elif property == 'tpsa':
                flat_reward = self.tpsa(mols)
            elif property == 'num_rings':
                flat_reward = self.count_num_rings(mols)
            elif property == 'sas':
                flat_reward = self.synthetic_assessibility(mols)
            elif property == 'qed':
                flat_reward = self.qed(mols)
                # print('reward2d.py flat_Reward qed ', flat_reward)
            elif property == 'LD50':
                # flat_reward = self.ld50(self.Y_scaler, self.task_model, mols)
                flat_reward = self.toxicity(self.task_model, mols)
                # print('reward2d.py flat_Reward ld50 ', flat_reward)
            elif property == 'Caco2':
                flat_reward = self.permeability(self.task_model, mols)
            elif property == 'Lipophilicity':
                flat_reward = self.lipo(self.task_model, mols)
            elif property == 'Solubility':
                flat_reward = self.sol(self.task_model, mols)
            elif property == 'BindingRate':
                flat_reward = self.bind(self.task_model, mols)
            elif property == 'MicroClearance':
                flat_reward = self.mclear(self.task_model, mols)
            elif property == 'HepatocyteClearance':
                flat_reward = self.hclear(self.task_model, mols)
            else:
                #property not known
                continue
            flat_rewards.append((flat_reward, property, self.cond_range[property][0], self.cond_range[property][2]))
        
        # print(f'reward_2d.py computing zinc radius for {len(mols)} mols')
        # import time
        # t0 = time.time()
        zinc_radius_flat_reward = self.searchAtomEnvironments_fraction(mols)
        zinc_radius_flat_reward_array = np.array(zinc_radius_flat_reward)
        if self.zinc_rad_scale:
            scaled_zinc_radius_flat_reward_array = zinc_radius_flat_reward_array*self.zinc_rad_scale
        else:
            scaled_zinc_radius_flat_reward_array = zinc_radius_flat_reward_array*1
        # print('finished ',time.time()-t0)
        if 'zinc_radius' in self.cond_range:
            flat_rewards.append((scaled_zinc_radius_flat_reward_array, "zinc_radius", None, None))
        # if 'qed' in self.cond_range:
        #     flat_rewards.append((self.qed(mols), "qed", None, None))
        return flat_rewards, zinc_radius_flat_reward

    def molecular_rewards(self, mols):
        '''Open endpoint to get all molecular rewards.
        '''
        flat_rewards, zinc_rad_flat = self._compute_flat_rewards(mols)
        cons_rew = self._consolidate_rewards(flat_rewards)
        if self.reward_aggregation == "add":
            agg = np.sum(cons_rew, axis=1)
            agg_rew_normal_fn = lambda x: scale_range(x, len(self.cond_range), 0, 1, 0)
            return cons_rew, flat_rewards, agg_rew_normal_fn(agg), zinc_rad_flat
        elif self.reward_aggregation == "mul":
            return cons_rew, flat_rewards, np.prod(cons_rew,axis = 1), zinc_rad_flat
        elif self.reward_aggregation == "add_mul":
            agg = np.sum(softplus(torch.Tensor(cons_rew)).numpy(), axis=1) + np.prod(cons_rew, axis=1)
            if self.zinc_rad_scale:
                # TODO: Reexamine zinc scaling implementation
                agg_rew_normal_fn = lambda x: scale_range(x, len(self.cond_range)*np.log(1+np.exp(1)) + np.log(1+np.exp(self.zinc_rad_scale))+self.zinc_rad_scale, 
                                                            (len(self.cond_range)+1)*np.log(2), 1, 0)   # len(self.cond_range)+1 : 6 cond_rewards + 1 zinc_radius reward
            else:
                agg_rew_normal_fn = lambda x: scale_range(x, 1+ (len(self.cond_range)+1)*np.log(1+np.exp(1)), 
                                                          (len(self.cond_range)+1)*np.log(2), 1, 0)   # len(self.cond_range)+1 : 6 cond_rewards + 1 zinc_radius reward

            return cons_rew, flat_rewards, agg_rew_normal_fn(agg), zinc_rad_flat