from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Crippen, DataStructs, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcFractionCSP3
from gflownet.utils.sascore import calculateScore
import numpy as np
import pandas as pd
from typing import List, Any, Optional, Set, Iterable
from functools import partial
from multiprocessing import Pool
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import scipy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from botorch.utils.multi_objective import infer_reference_point, pareto
    from botorch.utils.multi_objective.hypervolume import Hypervolume
except Exception as e:
    print('BOTorch Missing')
import torch
import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc import Evaluator
import random
import more_itertools as mit
from maplight import get_fingerprints

def is_molecule_successful(properties, ranges, preferences):
    """
    Determine if a molecule is successful based on multiple properties.

    Args:
        properties (list of float): List of molecular property values [p1, p2, ..., pk].
        ranges (list of tuple): List of desired ranges [(a1, b1), (a2, b2), ..., (ak, bk)].
        preferences (list of int): List of preferences for each property [-1, 0, 1].

    Returns:
        bool: True if the molecule is successful for all properties, False otherwise.
    """
    all_props_success = [False]*len(ranges)
    for i, (p, (a, b), preference) in enumerate(zip(properties, ranges, preferences)):
        if preference == -1:  # Lower values preferred
            lower_threshold = a * 0.9 if a > 0 else a * 1.1
            upper_threshold = a * 1.1 if a>0 else a *0.9
            if (p >= lower_threshold) and (p<=upper_threshold):
                all_props_success[i] = True
                continue
            else:
                return False
        elif preference == 1:  # Higher values preferred
            lower_threshold = b * 0.9 if b > 0 else b * 1.1
            upper_threshold = b * 1.1 if b > 0 else b * 0.9
            if (p >= lower_threshold) and (p<=upper_threshold):
                all_props_success[i] = True
                continue
            else:
                return False

        elif preference == 0:  # Range constraint
            lower_threshold = a * 0.9 if a > 0 else a * 1.1
            upper_threshold = b * 1.1 if b > 0 else b * 0.9
            if (p >= lower_threshold) and (p<=upper_threshold):
                all_props_success[i] = True
                continue
            else:
                return False
        else:
            raise ValueError("Preference must be -1, 0, or 1.")
    return all(all_props_success)

def success_percentage(molecules, ranges, preferences):
    """
    Calculate the success percentage for multiple molecules.

    Args:
        molecules (list of list of float): List of molecules, where each molecule is a list of property values.
        ranges (list of tuple): List of desired ranges [(a1, b1), (a2, b2), ..., (ak, bk)] for k properties.
        preferences (list of int): List of preferences for each property [-1, 0, 1].

    Returns:
        float: Success percentage as the fraction of successful molecules multiplied by 100.
    """
    successful_count = 0
    
    for properties in molecules:
        if is_molecule_successful(properties, ranges, preferences):
            successful_count += 1
    
    success_percentage = (successful_count / len(molecules)) * 100
    return success_percentage

def get_hypervolume(flat_rewards: torch.Tensor, zero_ref=True) -> float:
    """Compute the hypervolume of a set of trajectories.
    Parameters
    ----------
    flat_rewards: torch.Tensor
      A tensor of shape (num_trajs, num_of_objectives) containing the rewards of each trajectory.
    """
    # Compute the reference point
    if zero_ref:
        reference_point = torch.zeros_like(flat_rewards[0])
    else:
        reference_point = infer_reference_point(flat_rewards)
    # Compute the hypervolume
    hv_indicator = Hypervolume(reference_point)  # Difference
    return hv_indicator.compute(flat_rewards)

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint

def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps

def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()

def remove_duplicates(list_with_duplicates):
    """
    Borrowed from Guacamol
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set: Set[Any] = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list

def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Borrowed from Guacamol
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None

def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Borrowed from Guacamol
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)

def count_5_and_6_membered_rings(mol):
    ring_info = mol.GetRingInfo()
    num_five_six_membered_rings = len([ring for ring in ring_info.AtomRings() if (len(ring) == 6) or (len(ring) == 5) ])
    return num_five_six_membered_rings

def calculate_fingerprint(mol):
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def searchAtomEnvironments_fraction( mols, atomenv_dictionary, radius=2):
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
                    if k in atomenv_dictionary:
                        atomenvs += 1
        return atomenvs / max(mol.GetNumAtoms(), 1)
    return [per_mol_fraction(mol,radius)+np.finfo(np.float32).eps for mol in mols] # add \epsilon to prevent 0 composite reward upon multiplication of all flat rewards


# calculate success percentage for a single property
def calculate_success_for_property(prop_val, min_val, max_val, preference, relative_success=False):
    if relative_success:
        if preference == -1:
            if min_val>0:
                return 1 if prop_val<=min_val+0.1*(max_val-min_val) else 0
        elif preference == 1:
            return 1 if prop_val >= max_val - 0.1*(max_val-min_val) else 0
        else: 
            return 1 if min_val <= prop_val <= max_val else 0
    else:
        if preference == -1:  # Lower values preferred
            lower_threshold = min_val * 0.9 if min_val > 0 else min_val * 1.1
            return 1 if prop_val <= lower_threshold else 0
        elif preference == 1:  # Higher values preferred
            upper_threshold = max_val * 1.1 if max_val > 0 else max_val * 0.9
            return 1 if prop_val >= upper_threshold else 0
        else:  # Range constraint (preference == 0)
            # Adjust range bounds based on 10% tolerance
            lower_threshold = min_val * 0.9 if min_val > 0 else min_val * 1.1
            upper_threshold = max_val * 1.1 if max_val > 0 else max_val * 0.9
            return 1 if lower_threshold <= prop_val <= upper_threshold else 0
    

# calculate Tanimoto similarity between two fingerprints
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)



def calculate_success_percentage(smiles_list, task_conditionals, all_conditionals, predictor = None):
    ranges = [all_conditionals[r][0] for r in all_conditionals ]
    preferences = [all_conditionals[r][-1] for r in all_conditionals ]
    mols_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    conditional_keys = list(all_conditionals.keys())

    # Calculate properties for each molecule
    properties_list = [
        calculate_properties(mol, all_conditionals, predictor) 
        for mol in mols_list
    ]

    # Extract molecule properties in the same order as `conditional_keys`
    molecule_props = [
        [prop[key] for key in conditional_keys] for prop in properties_list
    ]
    return success_percentage(molecule_props, ranges, preferences)

def calculate_unique_ratio(smiles_list):
    unique_molecules = canonicalize_list(smiles_list, include_stereocenters=False)
    unique_ratio = len(unique_molecules) / len(smiles_list)
    return unique_ratio

def calculate_nmodes(smiles_list, rewards, tanimoto_thresh, reward_threshold):
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprints = [calculate_fingerprint(mol) for mol in mols_list]
    mol_data = [(smiles, reward, fingerprint) for smiles, reward, fingerprint in zip(smiles_list, rewards, fingerprints) if fingerprint is not None]
    mol_data.sort(key=lambda x: x[1], reverse=True)
    modes = []
    for smiles, reward, fingerprint in mol_data:
        if reward < reward_threshold:
            continue
        is_new_mode = True
        for mode_smiles, mode_fp in modes:
            similarity = tanimoto_similarity(fingerprint, mode_fp)
            if similarity > tanimoto_thresh:
                is_new_mode = False
                break
        if is_new_mode:
            modes.append((smiles, fingerprint))
    return len(modes)

def calculate_diversity(smiles_list):
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprints = [calculate_fingerprint(mol) for mol in mols_list]
    num_mols = len(mols_list)
    tanimoto_similarities = np.zeros((num_mols, num_mols))
    for i in range(num_mols):
        for j in range(i + 1, num_mols):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            tanimoto_similarities[i, j] = similarity
            tanimoto_similarities[j, i] = similarity
    avg_similarity = np.mean(tanimoto_similarities[np.triu_indices(num_mols, k=1)])
    diversity = 1 - avg_similarity
    return diversity

def calculate_fast_diversity(smiles_list):
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fingerprints = [calculate_fingerprint(mol) for mol in mols_list]
    return 1 - np.concatenate([Chem.DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i+1:]) for i in range(len(mols_list) - 1)]).mean()

def zinc_frag_novelty(smiles_list, atomenv_dictionary_path):
    with open(atomenv_dictionary_path, "rb") as f:
        atomenv_dictionary = pickle.load(f)
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    zinc_scores = searchAtomEnvironments_fraction(mols_list, atomenv_dictionary)
    novelty_scores = [score if score < 0.95 else 0 for score in zinc_scores]
    return np.average(novelty_scores)

def calculate_normalized_l1_diff(smiles_list, conditionals, predictor = None):
    '''predictor: test_molopt.TaskPredictor object, needed for TDC tasks
    '''
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    properties_list = [calculate_properties(mol, conditionals, predictor) for mol in mols_list]
    conditionals_l1 = {}
    for prop in conditionals:
        l1sum = 0
        c_low, c_high = conditionals[prop][0][0], conditionals[prop][0][1]
        preference = conditionals[prop][-1]
        for i in range(len(mols_list)):
            prop_val = properties_list[i][prop]
            if preference == 0:
                if (prop_val < c_low) or (prop_val > c_high):
                    l1sum += min(abs(prop_val - c_low), abs(prop_val - c_high))
            elif preference == 1:
                l1sum += abs(prop_val - c_high)
            elif preference == -1:
                l1sum += abs(prop_val - c_low)
        conditionals_l1[prop] = l1sum / ((i + 1) * (c_high - c_low))
    return conditionals_l1


def binding_evaluate(protein, smiles, train_smiles, minimized_docking_scores):
    df = pd.DataFrame()
    num_mols = len(smiles)
    df['smiles'] = smiles
    validity = len(df) / num_mols
    df['MOL'] = [Chem.MolFromSmiles(s) for s in smiles]

    train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    
    if 'sim' not in df.keys():
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['MOL']]
        train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
        max_sims = []
        for i in range(len(gen_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], train_fps)
            max_sims.append(max(sims))
        df['sim'] = max_sims



    uniqueness = len(set(df['smiles'])) / len(df)
    novelty = len(df[df['sim'] < 0.4]) / len(df)


    df[protein] = (np.array(minimized_docking_scores)/-1).tolist() 
    df['qed'] = [QED.qed(m) for m in df['MOL']]
    df['sa'] = [(10 - calculateScore(m)) / 9 for m in df['MOL']]

    df = df.drop_duplicates(subset=['smiles'])
    del df['MOL']

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr =  8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    else: raise ValueError('Unrecognized target protein')

    df = df[df[f'{protein}']!=0]
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]

    hit_len_df = len(df[df[protein] > hit_thr]) / len(df) 
    hit_num_mols = len(df[df[protein] > hit_thr]) / num_mols


    nov_df = df[df['sim'] < 0.4]
    nov_df = nov_df.sort_values(by=[protein], ascending=False)

    num_top5 = int(num_mols * 0.05)

    top_ds = nov_df.iloc[:num_top5][protein].mean(), nov_df.iloc[:num_top5][protein].std()
    nov_hit_len_df = len(nov_df[nov_df[protein] > hit_thr]) / len(df) 
    nov_hit_num_mols = len(nov_df[nov_df[protein] > hit_thr]) / num_mols
    
    return {'validity': validity, 'uniqueness': uniqueness,
            '0.4Tanimoto_novelty': novelty, 'top_ds': top_ds, 'hit': hit_len_df, 'hit_num_mols': hit_num_mols, 'nov_hit_len_df':nov_hit_len_df, 'nov_hit_num_mols':nov_hit_num_mols}

def get_metrics(generated_smiles_list, gen_smi_overall_rew, pretrain_conditionals, pretrain_smiles_list, task_conditionals, 
                predictor = None, gen_smi_moo_rew = None, hypervolume = True, task_offline_df=None, binding=False, protein=None, docking_scores=None):
    '''res_dict_path: path to dfictionary with sampled smiles and corresponding rewards
        predictor: test_molopt.TaskPredictor object, needed for TDC tasks
    '''
    evaluator = Evaluator(name = 'Diversity')
    metrics= {}
    if task_offline_df:
        task_offline_smiles = task_offline_df.smiles.values
        training_smiles = pretrain_smiles_list+task_offline_smiles
    else:
        training_smiles = pretrain_smiles_list
    all_conditionals = pretrain_conditionals.copy()
    all_conditionals.update(task_conditionals)
    gen_mols = [Chem.MolFromSmiles(smiles) for smiles in generated_smiles_list]
    gen_df = pd.DataFrame(gen_mols,columns=['MOL'])
    metrics['l1dist']= calculate_normalized_l1_diff(generated_smiles_list, all_conditionals, predictor)
    metrics['uniqueness'] = calculate_unique_ratio(generated_smiles_list)
    metrics['success%'] = calculate_success_percentage(generated_smiles_list, task_conditionals, all_conditionals, predictor=predictor)#, relative_success=False)
    metrics['diversity'] = calculate_fast_diversity(generated_smiles_list)
    metrics['tdc div']= evaluator(generated_smiles_list)
    metrics['novelty'] = guacamol_novelty(generated_smiles_list,training_smiles)
    metrics['#modes'] = calculate_nmodes(generated_smiles_list, gen_smi_overall_rew, tanimoto_thresh=0.5, reward_threshold=0.5)
    metrics['#scaffold']= bemisMurcoScaffold(generated_smiles_list)
    metrics['validity'] = len(gen_mols)/len(generated_smiles_list)
    metrics['#Circles'] = get_ncircle(gen_df)
    metrics['top-K'] = top_k_scores(gen_smi_overall_rew, generated_smiles_list)
    # metrics['IntDiv1'] = internal_diversity(generated_smiles_list)
    if hypervolume:
        metrics['Hypervolume'] = get_hypervolume(gen_smi_moo_rew)
    if binding:
        metrics['binding'] = binding_evaluate(protein, generated_smiles_list, training_smiles, docking_scores)
    return metrics

def top_k_scores(gen_smi_overall_rew, generated_smiles_list):
    top_smi = sorted(zip(gen_smi_overall_rew, generated_smiles_list), reverse=True)
    top_k_avg_rewards = {}
    top_k_values = [1, 10, 100]
    for k in top_k_values:
        top_k_tuples = top_smi[:k]
        avg_reward = sum(tup[0] for tup in top_k_tuples) / len(top_k_tuples)
        top_k_avg_rewards[f"top-{k}"] = avg_reward
    return top_k_avg_rewards
    

def guacamol_novelty(smiles_list, training_smiles):
    training_set_molecules = set(canonicalize_list(training_smiles, include_stereocenters=False))
    unique_molecules = canonicalize_list(smiles_list, include_stereocenters=False)
    novel_molecules = set(unique_molecules).difference(training_set_molecules)
    novel_ratio = len(novel_molecules) / len(smiles_list)
    return novel_ratio

class TaskPredictor:
    def __init__(self, task_model_paths):
        """
        Initialize the predictor with paths to models for each task.

        :param task_model_paths: A dictionary mapping task names to model file paths.
        """
        self.task_models = {}
        self.Y_scalers = {}
        for task, path in task_model_paths.items():
            with open(path, 'rb') as f:
                self.task_models[task], self.Y_scalers[task] = pickle.load(f)
                
    def scale_range( self, OldValue, OldMax, OldMin, NewMax, NewMin):
        OldRange = (OldMax - OldMin)  
        NewRange = (NewMax - NewMin)  
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        return NewValue

    def predict(self, task: str, smiles_list: List[str]):
        """
        Perform predictions for a specified task using the corresponding model and scaler.

        :param task: Name of the task (e.g., 'LD50', 'caco2').
        :param smiles_list: List of Smiles.
        :return: Normalized predictions for the specified task.
        """
        if task not in self.task_models:
            raise ValueError(f"Task '{task}' is not recognized or model is not loaded.")

        task_model = self.task_models[task]
        Y_scaler = self.Y_scalers[task]
        # smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        X_test = get_fingerprints(pd.Series(smiles_list))
        y_pred = Y_scaler.inverse_transform(task_model.predict(X_test, thread_count=1)).reshape(-1, 1)

        # Define normalization ranges for each task
        normalization_ranges = {
            'LD50': (6, 0),
            'Caco2': (-3, -8),
            'Lipophilicity': (5, -2),
            'Solubility': (2, -13),
            'BindingRate': (100, 0),
            'MicroClearance': (200, 0),
            'HepatocyteClearance': (200, 0),
        }

        if task not in normalization_ranges:
            raise ValueError(f"No normalization range defined for task '{task}'.")

        ypred1_normal_fn = lambda x: self.scale_range(x, *normalization_ranges[task], 1, 0)
        y_pred_normalized = ypred1_normal_fn(np.array(y_pred))
        return y_pred_normalized, y_pred #y_pred is true score


# SATURN Metrics

class NCircles():
    def __init__(self, threshold=0.75):
        super().__init__()
        self.sim_mat_func = similarity_matrix_tanimoto
        self.t = threshold
    
    def get_circles(self, args):
        vecs, sim_mat_func, t = args
        
        circs = []
        for vec in vecs:
            if len(circs) > 0:
                dists = 1. - sim_mat_func([vec], circs)
                if dists.min() <= t: continue
            circs.append(vec)
        return circs

    def measure(self, vecs, n_chunk=64):
        for i in range(3):
            vecs_list = [list(c) for c in mit.divide(n_chunk // (2 ** i), vecs)]
            args = zip(vecs_list, 
                       [self.sim_mat_func] * len(vecs_list), 
                       [self.t] * len(vecs_list))
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs)

def get_ncircle(df):
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['MOL']]
    return NCircles().measure(df['FPS'])


def bemisMurcoScaffold(smiles_list):
    # Set to store unique scaffolds
    unique_scaffolds = set()

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Get the Bemis-Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            # Convert scaffold to SMILES to check uniqueness
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            unique_scaffolds.add(scaffold_smiles)

    # Calculate chemical space coverage
    coverage = len(unique_scaffolds)

    return coverage


# calculate molecular properties
def calculate_properties(mol, all_conditionals, predictor=None):
    '''predictor: test_molopt.TaskPredictor object, needed for TDC tasks
    '''
    properties = {}
    if "num_rings" in all_conditionals:
        properties["num_rings"] = count_5_and_6_membered_rings(mol)
    if "tpsa" in all_conditionals:
        properties["tpsa"] = Descriptors.TPSA(mol)
    if "qed" in all_conditionals:
        properties["qed"] = QED.qed(mol)
    if "sas" in all_conditionals:
        properties["sas"] = calculateScore(mol)
    if "Mol_Wt" in all_conditionals:
        properties["Mol_Wt"] = Descriptors.MolWt(mol)
    if "logP" in all_conditionals:
        properties["logP"]= Crippen.MolLogP(mol)
    if "fsp3" in all_conditionals:
        properties["fsp3"] = CalcFractionCSP3(mol)
    if "LD50" in all_conditionals:
        properties["LD50"] = predictor.predict("LD50", [Chem.MolToSmiles(mol)])[1][0][0]
    if "Caco2" in all_conditionals:
        properties["Caco2"] = predictor.predict("Caco2", [Chem.MolToSmiles(mol)])[1][0][0]
    if "BindingRate" in all_conditionals:
        properties["BindingRate"] = predictor.predict("BindingRate", [Chem.MolToSmiles(mol)])[1][0][0]
    if "Solubility" in all_conditionals:
        properties["Solubility"] = predictor.predict("Solubility", [Chem.MolToSmiles(mol)])[1][0][0]
    if "MicroClearance" in all_conditionals:
        properties["MicroClearance"] = predictor.predict("MicroClearance", [Chem.MolToSmiles(mol)])[1][0][0]
    if "HepatocyteClearance" in all_conditionals:
        properties["HepatocyteClearance"] = predictor.predict("HepatocyteClearance", [Chem.MolToSmiles(mol)])[1][0][0]
    if "Lipophilicity" in all_conditionals:
        properties["Lipophilicity"] = predictor.predict("Lipophilicity", [Chem.MolToSmiles(mol)])[1][0][0]
    return properties

#