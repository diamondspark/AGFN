import torch
from gflownet.utils.transforms import thermometer
import numpy as np
from scipy.stats import truncnorm
import random

class ConditionalInfo():
    def __init__(self, conditional_range_dict, cond_prop_var, num_thermometer_dim, OOB_percent=None) -> None:
        self.cond_range = conditional_range_dict
        self.num_thermometer_dim = num_thermometer_dim
        self.cond_prop_var = cond_prop_var
        self.OOB_prob = OOB_percent

    def get_scalar(self, n_trajs, bounds, bound_idx, int=False, v_lower=None):
        '''
        Helper method to get the scalar value for the lower and upper bounds
        of each property. Returns np.array of size (n_trajs,1) corresponding
        to lower/upper bound for all the mols in the batch for that property.
        params:
        bounds = [lbound, ubound]
        bound_idx = 0/1 for lbound and ubound respectively
        v_lower = unnormalized scalars of lbounds for n_trajs
                  Needed to restrict sampled scalars in v_upper 
                  to take values greater than the respective scalar
                  in v_lower. We could consider V_upper^j ~ U(min(V_lower^(j), ubound))
                  but iterating over entire batch for this expensive. Until a better
                  solution is found, restrict v_upper to sample conditionals for idx j
                  where corresponding jth idx of v_lower was sampled as well.
        '''
        value = bounds[bound_idx]
        temp_v = value*np.ones(n_trajs)
        mask = np.random.randint(0,2,size=temp_v.shape).astype(bool)
        # if bound_idx==0:
        samples = np.random.uniform(*bounds,*temp_v.shape)
        if int:
            samples = np.round(samples)
        # temp_v[mask]= samples[mask]
        if bound_idx==1:
            mask = mask*np.array(v_lower==bounds[0])
        temp_v[mask]= samples[mask]
        return temp_v

    def thermometer_encoding(self, cond_info, ft_conditionals_dict=None):
        '''
        Get thermometer encoding for the scalars in cond_info.
        Concatenate the encoding for each scalar i.e. for lbound and ubound
        '''
        encoding = []
        for property in cond_info.keys():
            lower_bound = ft_conditionals_dict[property][1][0] if ft_conditionals_dict else self.cond_range[property][1][0]
            upper_bound = ft_conditionals_dict[property][1][1] if ft_conditionals_dict else self.cond_range[property][1][1]
            for i in range(2):
                cond = cond_info[property][i].astype(np.float32)
                # print(f'{property} {cond}')
                encoding.append(thermometer(torch.tensor(cond), self.num_thermometer_dim,
                            lower_bound, upper_bound))
        encoding = torch.cat(encoding, dim=1) #use torch.cat for torch v1.10, for newer versions use torch.concatenate
        return encoding

    def compute_cond_info_forward(self, n_trajs, ft_conditionals_dict =None):
        ''' param: n_trajs = num_online
            ft_condtionals_dict: cond_range_dict for finetuning conditionals. 
        '''
        if ft_conditionals_dict:
            cond_info = dict()
            for property in ft_conditionals_dict.keys():
                if (property == 'zinc_radius') or( property =='qed'):
                    continue
                temp_scalar, temp_encoding =[],[]
                bounds = ft_conditionals_dict[property][0]
                if property == 'num_rot_bonds':
                    v_lower = self.get_scalar( n_trajs, bounds, 0, int=True)
                    v_upper = self.get_scalar( n_trajs, bounds, 1, int=True,v_lower= v_lower)
                else:
                    v_lower = self.get_scalar( n_trajs, bounds, 0, int=False)
                    v_upper = self.get_scalar( n_trajs, bounds, 1, int=False, v_lower= v_lower)
                    
                temp_scalar.append(v_lower)
                temp_scalar.append(v_upper)
                cond_info[property] = temp_scalar
            return cond_info
        else:
            cond_info = dict()
            # print('Moltask.py compute_cond_info cond_rangekeys ', self.cond_range.keys())
            for property in self.cond_range.keys():
                if (property == 'zinc_radius') or( property =='qed'):
                    continue
                temp_scalar, temp_encoding =[],[]
                bounds = self.cond_range[property][0]
                if property == 'num_rot_bonds':
                    v_lower = self.get_scalar( n_trajs, bounds, 0, int=True)
                    v_upper = self.get_scalar( n_trajs, bounds, 1, int=True,v_lower= v_lower)
                else:
                    v_lower = self.get_scalar( n_trajs, bounds, 0, int=False)
                    v_upper = self.get_scalar( n_trajs, bounds, 1, int=False, v_lower= v_lower)
                    
                temp_scalar.append(v_lower)
                temp_scalar.append(v_upper)
                cond_info[property] = temp_scalar
            return cond_info
    

    def compute_cond_info_backward(self, flat_rewards):
        ''' Samples a normal with mean = predicted property and predefined
            variance to create the range for this property for this mol.  
            flat_rewards: calculated values for each property for each mol in batch.
            ISSUE: We are calculating a new range for each property for each mol in dataset.
            That is O(m*n) calculation. Potentially slow. Can we do something cleverer?
        '''

 #        [(array([125.067, 129.04 , 131.201,  97.009]), 'mol_wt', [250, 500], -1),
 # (array([1., 0., 0., 0.]), 'fsp3', [0.2, 0.6], 1),
 # (array([-0.6714, -1.1418, -1.5289, -0.1493]), 'logP', [-5, 4], -1),
 # (array([3, 0, 2, 0]), 'num_rot_bonds', [0, 8], -1),
 # (array([ 67.31, 100.96,  64.07,  42.53]), 'tpsa', [0, 140], -1)]
        
        cond_bck_range= dict()
        oob_rand_samples = np.random.uniform(0,1,len(self.cond_range))
        for i, prop_name in enumerate(self.cond_range.keys()):
            if (prop_name == 'zinc_radius') or (prop_name == 'qed'):
                continue
            acceptable_min, acceptable_max = self.cond_range[prop_name][1][0], self.cond_range[prop_name][1][1]
            mean, std_dev = flat_rewards[i][0], self.cond_prop_var[prop_name]
            assert flat_rewards[i][1] == prop_name  

            if oob_rand_samples[i]<=self.OOB_prob:
                # do_custom_bounds()
                if random.random()<0.5:
                    #fix lower bound (range1) and sample upper bound (range2) from U(lowerbound, mol prop(x))
                    lower_bound = range1 = np.array([acceptable_min]*len(flat_rewards[i][0]))
                    upper_bound = range2 =  np.random.uniform(lower_bound,mean)
                    # print(f'moltask.py LB fixed {prop_name} , range2 {range2.shape}, range1 {range1.shape}')
                else:
                    #fix upper bound and sample lower bound from U(mol prop(x), upperbound)
                    upper_bound = range2 = np.array([acceptable_max]*len(flat_rewards[i][0]))
                    lower_bound = range1 =  np.random.uniform(mean, upper_bound)
                    # print(f'moltask.py UB fixed {prop_name}')
            else:
 
                range1 = truncnorm.rvs((acceptable_min - mean) / std_dev, (acceptable_max - mean) / std_dev, 
                                    loc=mean, scale = std_dev, size=len(flat_rewards[i][0]))
                range2 = truncnorm.rvs((acceptable_min - mean) / std_dev, (acceptable_max - mean) / std_dev, 
                                loc=mean, scale = std_dev, size=len(flat_rewards[i][0]))
                # print('MolTask.py range1 not OOB', range1.shape)
            
            if prop_name == 'num_rot_bonds' or prop_name == 'num_rings':
                #num_rot_bonds can only be int. Hence, rounding
                range1, range2 = np.round(range1), np.round(range2)
                
            prop_per_mol = []
            for j in range(len(range1)):
                prop_per_mol.append([min(range1[j], range2[j]),max(range1[j], range2[j])])

            # transpose prop_per_mol such that each property in cond_bck_range
            # contains a list : [arr(low for all mols), arr(high for all mols)]
            # By such a transpose, cond_bck_range becomes cond_info that can be used
            # as is to get thermometer encoding for back conditionals.
            tmp = np.array(prop_per_mol).T
            cond_bck_range[prop_name]= [tmp[0],tmp[1]]
            
        return cond_bck_range
# # Suggested ranges from 
# # https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/64e5333e3fdae147faa01202/original/stoplight-a-hit-scoring-calculator.pdf
# # First list is drug like range, second list is lower and upper bounds for thermometer encoding, +/-1 indicates if higher/lower is better.
# # Latter are more liberal bounds
# conditional_range_dict = {'mol_wt': [[250,500],[0,600], -1],           #[400, 500], lower is better, open lower bound
#                           'fsp3':[[0.2,0.6],[0,1], 1],            #[0.2,0.3], higher is better, open upper bound
#                           'logP':[[-5,4],[-5,5], -1],                #[2,3], #lower is better, open lower bound
#                           'num_rot_bonds':[[0,8],[0,15], -1],      #[7,10], #lower is better, open lower bound
#                           'tpsa':[[0,140],[0,200], -1],               #[120,140],  #lower is better, open lower bound
#                           #'water_solubility':[[10,100],[0,200]]  #[10,50] # larger the better, open upper bound #Not doing this as logS prediction is typically based on models 
#                           }
# cond_prop_var= {'mol_wt':50,
#                 'fsp3':0.1,
#                 'logP':2,
#                 'num_rot_bonds':2,
#                 'tpsa':2}
# conditional = ConditionalInfo(conditional_range_dict, 16)
# cond_info = conditional.compute_cond_info_forward(n_trajs = 4)
# encoding = conditional.thermometer_encoding(cond_info)

# flat_rewards = Rewards(conditional_range_dict).molecular_rewards(mols)[1]
# cond_info_bck = conditional.compute_cond_info_backward(flat_rewards)
# encoding_bck = conditional.thermometer_encoding(cond_info_bck)
# encoding_bck.shape