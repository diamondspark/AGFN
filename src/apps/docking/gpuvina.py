import os
import subprocess
import tempfile
from typing import List, Tuple
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import RDLogger
from rdkit.Chem import rdDistGeom
from useful_rdkit_utils import get_center
from tqdm import tqdm
import pickle
from pathlib import Path
import json
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Enable this with correct ID to let docking run on seperate GPU. Enables larger batch size training


logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

def gpu_vina_installed(vina_path):
    if os.path.exists(vina_path):
        return True
    return False

def read_pdbqt(fn):
    """
    Read a pdbqt file and return the RDKit molecule object.

    Args:
        - fn (str): Path to the pdbqt file.

    Returns:
        - mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
    """
    pdbqt_mol = PDBQTMolecule.from_file(fn, is_dlg=False, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    return rdkitmol_list[0]

def smile_to_conf(smile: str, n_tries=5) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)

    tries = 0
    while tries < n_tries:
        try:
            params = rdDistGeom.ETKDGv3()

            # set the parameters
            params.useSmallRingTorsions = True
            params.randomSeed = 0
            params.numThreads = 1

            # generate the conformer
            rdDistGeom.EmbedMolecule(mol, params)

            # add hydrogens
            mol = Chem.AddHs(mol, addCoords=True)

            if mol.GetNumConformers() > 0:
                return mol

            tries += 1
        except Exception as e:
            print(e, smile)
            mol = None

    print(f"Failed to generate conformer for {smile}")
    return mol

def mol_to_pdbqt(mol: Chem.Mol, pdbqt_file: str):

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    for setup in mol_setups:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(pdbqt_file, "w") as f:
                f.write(pdbqt_string)
            break
        else:
            print(f"Failed to write pdbqt file: {error_msg}")


def parse_affinty_from_pdbqt(pdbqt_file: str) -> float:
    with open(pdbqt_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "REMARK VINA RESULT" in line:
            return float(line.split()[3])
    return None


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(repo_root, "data/docking/")

with open(f'./data/docking/tmp_config.json','r') as f:
    TARGETS = json.load(f)

# TARGETS = {
#     "2bm2": {
#         "receptor": os.path.join(DATA_DIR, "2bm2/2bm2_protein.pdbqt"),
#         "center_x": 40.415,
#         "center_y": 110.986,
#         "center_z": 82.673,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 30,
#     },
#     "8azr": {
#         "receptor": os.path.join(DATA_DIR, "8azr/8azr.pdbqt"),
#         # "ref_ligand": os.path.join(DATA_DIR, "kras/8azr_ref_ligand.sdf"),
#         "center_x": 21.466,
#         "center_y": -0.650,
#         "center_z": 5.028,
#         "size_x": 18,
#         "size_y": 18,
#         "size_z": 18,
#         "num_atoms": 32,
#     },
#     "trmd": {
#         "receptor": os.path.join(DATA_DIR, "trmd/6qrd.pdbqt"),
#         "center_x": 16.957,
#         "center_y": 21.772,
#         "center_z": 33.296,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 34,
#     },
#      "fa7": {
#         "receptor": os.path.join(DATA_DIR, "fa7/fa7.pdbqt"),
#         "center_x": 10.131,
#         "center_y": 41.879,
#         "center_z": 32.097,
#         "size_x": 20.673,
#         "size_y": 20.198,
#         "size_z": 21.362,
#     },
#     "parp1": {
#         "receptor": os.path.join(DATA_DIR, "parp1/parp1.pdbqt"),
#         "center_x": 26.413,
#         "center_y": 11.282,
#         "center_z": 27.238,
#         "size_x": 18.521,
#         "size_y": 17.479,
#         "size_z": 19.995,
#     },
#     "5ht1b": {
#         "receptor": os.path.join(DATA_DIR, "5ht1b/5ht1b.pdbqt"),
#         "center_x": -26.602,
#         "center_y": 5.277,
#         "center_z": 17.898,
#         "size_x": 22.5,
#         "size_y": 22.5,
#         "size_z": 22.5,
#     },
#     "jak2": {
#         "receptor": os.path.join(DATA_DIR, "jak2/jak2.pdbqt"),
#         "center_x": 114.758,
#         "center_y": 65.496,
#         "center_z": 11.345,
#         "size_x": 19.033,
#         "size_y": 17.929,
#         "size_z": 20.283,
#     },
#     "braf": {
#         "receptor": os.path.join(DATA_DIR, "braf/braf.pdbqt"),
#         "center_x": 84.194,
#         "center_y": 6.949,
#         "center_z": -7.081,
#         "size_x": 22.032,
#         "size_y": 19.211,
#         "size_z": 14.106,
#     },
# }


class QuickVina2GPU(object):

    def __init__(
        self,
        vina_path: str,
        target: str = None,
        target_pdbqt: str = None,
        reference_ligand: str = None,
        input_dir: str = None,
        out_dir: str = None,
        save_confs: bool = False,
        reward_scale_max: float = -1.0,
        reward_scale_min: float = -10.0,
        thread: int = 8000, #1000,
        print_time: bool = False,
        print_logs: bool = False,
    ):
        """
        Initializes the QuickVina2GPU class with configuration for running QuickVina 2 on GPU.

        Give either a code for a target or a PDBQT file.

        Args:
            - vina_path (str): Path to the Vina executable.
            - target (str, optional): Target identifier. Defaults to None.
            - target_pdbqt (str, optional): Path to the target PDBQT file. Defaults to None.
            - reference_ligand (str, optional): Path to the reference ligand file. Defaults to None.
            - input_dir (str, optional): Directory for input files. Defaults to a temporary directory.
            - out_dir (str, optional): Directory for output files. Defaults to None, will use input_dir + '_out'.
            - save_confs (bool, optional): Whether to save conformations. Defaults to False.
            - reward_scale_max (float, optional): Maximum reward scale. Defaults to -1.0.
            - reward_scale_min (float, optional): Minimum reward scale. Defaults to -10.0.
            - thread (int, optional): Number of threads to use. Defaults to 8000.
            - print_time (bool, optional): Whether to print execution time. Defaults to True.

        Raises:
        - ValueError: If the target is unknown.

        """
        self.vina_path = vina_path
        self.save_confs = save_confs
        self.thread = thread
        self.print_time = print_time
        self.print_logs = print_logs
        self.reward_scale_max = reward_scale_max
        self.reward_scale_min = reward_scale_min

        if target is None and target_pdbqt is None:
            raise ValueError("Either target or target_pdbqt must be provided")

        if input_dir is None:
            input_dir = tempfile.mkdtemp()
        
        self.input_dir = input_dir
        Path(self.input_dir).mkdir(parents=True, exist_ok=True)
        self.out_dir = input_dir + "_out"

        if target.lower() in TARGETS:
            self.target_info = TARGETS[target.lower()]
        else:
            raise ValueError(f"Unknown target: {target}")

        for key, value in self.target_info.items():
            setattr(self, key, value)

    def _write_config_file(self):

        config = []
        config.append(f"receptor = {self.receptor}")
        config.append(f"ligand_directory = {self.input_dir}")
        config.append(f"opencl_binary_path = /groups/cherkasvgrp/Vina-GPU-2.1/QuickVina2-GPU-2.1/") #{VinaConfig.opencl_binary_path}")
        config.append(f"center_x = {self.center_x}")
        config.append(f"center_y = {self.center_y}")
        config.append(f"center_z = {self.center_z}")
        config.append(f"size_x = {self.size_x}")
        config.append(f"size_y = {self.size_y}")
        config.append(f"size_z = {self.size_z}")
        config.append(f"thread = {self.thread}")

        with open(os.path.join(self.input_dir, "../config.txt"), "w") as f:
            f.write("\n".join(config))
        # print(os.path.join(self.input_dir, "../config.txt"))
        # print(config)

    def _write_pdbqt_files(self, smiles: List[str]):

        # Convert smiles to mols
        mols = [smile_to_conf(smile, n_tries=1) for smile in tqdm(smiles, desc="Smiles conformation calculation")]

        
        # Remove None
        # mols = [mol for mol in mols if mol is not None]

        # Write pdbqt files
        for i, mol in enumerate(mols):
            pdbqt_file = os.path.join(self.input_dir, f"input_{i}.pdbqt")
            try:
                mol_to_pdbqt(mol, pdbqt_file)
            except Exception as e:
                print(f"Failed to write pdbqt file: {e}")

    def _teardown(self):
        # Remove input files
        for file in os.listdir(self.input_dir):
            os.remove(os.path.join(self.input_dir, file))
        os.rmdir(self.input_dir)

        # Remove output files
        if os.path.exists(self.out_dir):
            for file in os.listdir(self.out_dir):
                os.remove(os.path.join(self.out_dir, file))
            os.rmdir(self.out_dir)

    def _run_vina(self):

        result = subprocess.run(
            [self.vina_path, "--config", os.path.join(self.input_dir, "../config.txt")], capture_output=True, text=True
        )
        if self.print_time:
            print(result.stdout.split("\n")[-2])
        if self.print_logs:
            print(result.stdout.split("\n"))

        if result.returncode != 0:
            print(f"Vina failed with return code {result.returncode}")
            print(result.stderr)
            return False

    def _parse_results(self):

        results = []
        failed = 0

        for i in range(self.batch_size):
            pdbqt_file = os.path.join(self.out_dir, f"input_{i}_out.pdbqt")
            if os.path.exists(pdbqt_file):
                affinity = parse_affinty_from_pdbqt(pdbqt_file)
            else:
                affinity = 0.0
                failed += 1
            results.append((affinity))

        if failed > 0:
            print(f"WARNING: Failed to calculate affinity for {failed}/{self.batch_size} molecules")

        return results

    def _parse_docked_poses(self):
        poses = []
        failed = 0

        for i in range(self.batch_size):
            pdbqt_file = os.path.join(self.out_dir, f"input_{i}_out.pdbqt")
            if os.path.exists(pdbqt_file):
                mol = read_pdbqt(pdbqt_file)
                poses.append(mol)
            else:
                poses.append(None)
                failed += 1

        if failed > 0:
            print(f"WARNING: Failed to read docked pdbqt files for {failed}/{self.batch_size} molecules")

        return poses

    def _check_outputs(self):
        if not os.path.exists(self.out_dir):
            return False
        return True

    def calculate_rewards(self, smiles: List[str]) -> List[Tuple[str, float]]:

        self.batch_size = len(smiles)
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]

        # Write input files, config file and run vina
        self._write_pdbqt_files(smiles)
        self._write_config_file()
        self._run_vina()

        # Parse results
        if self._check_outputs():
            affinties = self._parse_results()
        else:
            affinties = [0.0] * self.batch_size

        # Scale affinities to calculate rewards
        affinties = np.array(affinties)
        rewards = (affinties + self.reward_scale_min) / (self.reward_scale_min + self.reward_scale_max) - 1

        print(
            f"AFFINITIES: mean={round(np.mean(affinties), 3 )}, std={round(np.std(affinties), 3)}, min={round(np.min(affinties), 3)}, max={round(np.max(affinties), 3)}"
        )


        return smiles, list(affinties), list(rewards)

    def dock_mols(self, smiles: List[str]) -> List[Tuple[str, float]]:
        self.batch_size = len(smiles)

        # Write input files, config file and run vina
        self._write_pdbqt_files(smiles)
        self._write_config_file()
        self._run_vina()

        # Parse results
        affinties = self._parse_results()

        # Scale affinities to calculate rewards
        affinties = np.array(affinties)
        mols = self._parse_docked_poses()

        print(
            f"AFFINITIES: mean={round(np.mean(affinties), 3 )}, std={round(np.std(affinties), 3)}, min={round(np.min(affinties), 3)}, max={round(np.max(affinties), 3)}"
        )

        return mols, affinties



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_name', type=str)
    parser.add_argument('gfn_samples_path', type=str)
    parser.add_argument('vina_path', type=str)
    args = parser.parse_args()

    vina = QuickVina2GPU(vina_path=args.vina_path, target=args.target_name)
    with open(args.gfn_samples_path+'/sampled_mols.pkl', 'rb') as f:
        mols = pickle.load(f)

    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    outs = vina.calculate_rewards(smiles_list) 
    with open(f'{args.gfn_samples_path}/{args.target_name}_docked.pkl','wb') as f:
        pickle.dump(outs,f)
        
    os.remove(args.gfn_samples_path+'/sampled_mols.pkl')
    # print('gpu_vina.py docking finished')
    