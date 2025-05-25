# Atomic GFlowNets- Pretraining & Finetuning atom based GFlowNets with Inexpensive Rewards for Molecule Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)[![Paper](https://img.shields.io/badge/arXiv-2503.06337-b31b1b.svg)](https://arxiv.org/abs/2503.06337)

## 📄 Paper Reference

**Title**: *Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation*

**Authors**: Mohit Pandey, Gopeshh Subbaraj, Artem Cherkasov, Martin Ester, Emmanuel Bengio

**arXiv**: [arXiv:2503.06337](https://arxiv.org/abs/2503.06337)

This repository provides code and configurations to replicate the pretraining, fine-tuning, and case-studies (denovo molecule design & lead optimization) experiments reported in our paper, including support for both standard Trajectory Balance and Relative Trajectory Balance (RTB).

## ⚙️ Installation

```bash
conda create -n agfn python=3.8.20
conda activate agfn
cd AGFN
pip install -r requirements_dev.txt
pip install -e .
```

## 🔬 Pretraining

To begin pretraining the AGFN model, follow these steps:

```bash
cd ./AGFN
conda activate agfn
python ./src/pretrainSrc/driver.py
```

This uses the default training configuration. To modify default settings update `./src/config/pretrain.yml`.

⚠️ Note:
By default, the script utilizes **all available GPUs** on the node. To restrict the number of GPUs, update the `world_size` parameter in the `if __name__ == '__main__'` block of `driver.py`.

The code has been tested on up to **8 NVIDIA A100 GPUs**.

Model checkpoints will be saved to: `./AGFN_logs/[wandb_run_name]/*.pt`. All configurable pretraining hyperparameters are located in: `.agfn/config/pretrain.yml`

### 🧠 Pretrained Weights

We provide pretrained weights for **AGFN-large**
(Trained on 6.75 million Enamine compounds, ~9.36 million parameters)

📥 [Download pretrained model](https://drive.google.com/file/d/1BmHF7gskOIKkFLn5KA0gfrdvxf2yNrY7/view?usp=sharing)

Place the downloaded checkpoint file at ./saved_models/pretrained/[pretrained_model].pt

🔜 Weights for smaller AGFN models will be released soon.

## 🧪 Finetuning

Similar to the pretraining setup, the tunable hyperparameters for finetuning are placed in:
`./src/config/finetune.yml`

The following fields of finetune.yml should be sufficient to recreate the experiments reported in our paper:

•	type: 'finetuning' / 'rtb' for vanilla finetuning with Trajectory Balance or Relative Trajectory Balance (RTB)

•	objective: property_constrained_optimization / property_targeting / property_optimization (We expect most users to be interested in property_constrained_optimization. See the Experiments section of our paper for more details.)

•	**subtype**: preserved / DRA (Dynamic Range Adjustment)

•	**task**: currently supports the following tasks out of the box:

◦	`Caco2`

◦	`LD50`

◦	`Lipophilicity`

◦	`Solubility`

◦	`BindingRate`

◦	`MicroClearance`

◦	`HepatocyteClearance`

•	**task_possible_range**: Refer to Table 8 and 9 in the appendix of our paper.

•	**pref_dir**: Preference direction; Refer to Table 9 in the appendix.

•	**task_model_path**: Path to the Maplight model for the task (e.g. ./saved_models/task_models/modelname.pt)

•	**offline_data**: True / False; whether to finetune with hybrid offline + online data or purely online

•	**offline_df_path**: Path to offline data used when offline_data == True; refer to files in ./data/task_files/ for formatting

•	**saved_model_path**: Path to pretrained AGFN prior

🔜 Support for custom tasks is coming soon!

### 🖥️ Multi-GPU Usage

By default, the script utilizes all available GPUs on the node. To restrict the number of GPUs used during training, update the world_size parameter in the `if __name__ == '__main__'` block of `ft_driver.py`.

### 🚀 Running Fine-Tuning

To run fine-tuning:

```bash
python ./src/finetuneSrc/ft_driver.py ./src/config/finetune.yml
```

## 💊 Sampling Molecules

To sample molecules using a fine-tuned GFlowNet model, run:

```bash
python ./src/agfn/sampling.py [finetuned_model_path] [n_samples] --bs [batch_size]
```

### 🔧 Arguments:

•	finetuned_model_path: Path to your trained .pt model file.
•	n_samples: Total number of SMILES to sample.
•	--bs: (Optional) Batch size used during sampling. Default is 32.

### 📁 Output:

•	Sampled SMILES will be saved to: ./data/gfn_samples/smiles_checkpoints/
•	If n_samples > 1000, intermediate checkpoints (10%, 20%, ..., 90%) will be saved incrementally.
•	The final SMILES list (100%) is always saved as smiles_final.pkl.

## 🔬 Applications

### 1. De Novo Design of Target-Specific Binders with Molecular Docking

This pipeline enables the generation of de novo molecules for a target protein, with molecular docking evaluation using **QuickVina2** as rewards.

### 🔧 Configuration

Update the following fields in `./config/denovo.yml`:

- **`target_name`**: Specifies the docking target. Supported targets out of the box:

  - `"5ht1b"`
  - `"fa7"`
  - `"parp1"`
  - `"jak2"`
  - `"braf"`
- **`saved_model_path`**:
  Path to the pretrained AGFN prior model.
- **`vina_path`**:
  Path to your local installation of QuickVina2.
- For a **custom target**, create an entry under the `target_grid` field.`target_grid` is a dictionary where:

  - The **key** is your custom target name.
  - The **value** is another dictionary with QuickVina docking parameters.

  🔍 Refer to existing targets in `denovo.yml` for formatting examples.

  📌 Additionally, place the prepared receptor file for the custom target at: `./data/docking/\custom\_target.pdbqt` You can prepare `.pdbqt` files using **AutoDockTools**. For example, see: [prepare_receptor4.py](https://github.com/sahrendt0/Scripts/blob/master/docking_scripts/prepare_receptor4.py)

Other hyperparameters can be left at their default values or customized based on your use case.

### 🚀 Running Training

```bash
python ./src/apps/docking/denovo/denovo_driver.py ./src/config/denovo.yml
````

> ⚠️ **Note**:
> This setup is optimized for a **2-GPU** configuration.
> For **single-GPU** setups, if you encounter docking-related CUDA errors, consider **reducing** the `training_batch_size` in `denovo.yml`.

### Sampling proceeds similarly to sampling molecules for fine-tuning.

## 📖 Citation

If you use this codebase or find it helpful in your research, please cite our paper:

```bibtex
@misc{pandey2025pretraininggenerativeflownetworks,
      title={Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation}, 
      author={Mohit Pandey and Gopeshh Subbaraj and Artem Cherkasov and Martin Ester and Emmanuel Bengio},
      year={2025},
      eprint={2503.06337},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06337}, 
}
```

## 📬 Contact

For questions or issues, please open an issue in the GitHub repository or contact the authors listed in the paper.
