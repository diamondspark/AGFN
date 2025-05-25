from typing import Any, Dict
from gflownet.models.graph_transformer import GraphTransformerGFN
import copy
import torch
import ast
import pathlib
import os
import json
# import git
from torch import Tensor
import warnings
warnings.filterwarnings("ignore")


class ModelSaveScheduler:
    def __init__(self, start_interval=50, max_interval=500, growth_factor=1.2):
        """
        Initializes the scheduler.
        
        Parameters:
        - start_interval: Initial interval (in steps) for saving the model.
        - max_interval: Maximum interval (in steps) after which the saving frequency should plateau.
        - growth_factor: The factor by which the interval increases.
        """
        self.current_interval = start_interval
        self.max_interval = max_interval
        self.growth_factor = growth_factor
        self.steps_since_last_save = 0
    
    def should_save(self):
        """
        Determines whether the model should be saved at the current step.
        
        Parameters:
        - step: The current step in training.
        
        Returns:
        - bool: True if the model should be saved, False otherwise.
        """
        self.steps_since_last_save += 1
        
        if self.steps_since_last_save >= self.current_interval:
            # Save the model
            self.steps_since_last_save = 0  # Reset the counter
            
            # Increase the interval, but cap it at max_interval
            self.current_interval = min(int(self.current_interval * self.growth_factor), self.max_interval)
            return True
        
        return False


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], algo, rng, device: torch.device, env, ctx,):
        self.env = env
        self.ctx = ctx
        self.hps = hps
        self.rng = rng
        self.device = device
        self.algo = algo
        self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"],
                                         do_bck=hps['tb_p_b_is_parameterized'], num_graph_out=int(self.hps["tb_do_subtb"]), num_mlp_layers=self.hps['num_mlp_layers'])

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(
            non_Z_params,
            hps["learning_rate"],
            (hps["momentum"], 0.999),
            weight_decay=hps["weight_decay"],
            eps=hps["adam_eps"],
        )
        self.opt_Z = torch.optim.Adam(Z_params, hps["Z_learning_rate"], (0.9, 0.999))
        if hps["lr_decay"]:
            self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / hps["lr_decay"]))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2 ** (-steps / hps["Z_lr_decay"]))

        self.sampling_tau = hps["sampling_tau"]
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model
        eps = hps["tb_epsilon"]
        hps["tb_epsilon"] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.clip_grad_param = hps["clip_grad_param"]
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            "none": (lambda x: None),
        }[hps["clip_grad_type"]]
        self.ckpt_scheduler = ModelSaveScheduler()

        os.makedirs(self.hps["log_dir"], exist_ok=True)
        fmt_hps = "\n".join([f"{f'{k}':40}:\t{f'({type(v).__name__})':10}\t{v}" for k, v in sorted(self.hps.items())])
        with open(pathlib.Path(self.hps["log_dir"]) / "hps.json", "w") as f:
            json.dump(self.hps, f)

    def step(self, loss: Tensor):
        loss.backward()
        if self.hps['clip_grad']:    
            for i in self.model.parameters():
                self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        if self.hps['lr_decay']:
            self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

    def _save_state(self, it, run_name):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict()],
                "hps": self.hps,
                "step": it,
                'opt': self.opt.state_dict(),
            },
            open(pathlib.Path(self.hps["log_dir"]) / f"model_state_{run_name}_{it}.pt", "wb"),
        )
        
class GFNTrainerRTB(GFNTrainer):
    def __init__(self, hps: Dict[str, Any], algo, rng, device: torch.device, env, ctx):
        super().__init__(hps, algo, rng, device, env, ctx)
        self.model_prior = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"],
                                         do_bck=hps['tb_p_b_is_parameterized'], num_graph_out=int(self.hps["tb_do_subtb"]), num_mlp_layers=self.hps['num_mlp_layers'])
                                        #  ,
                                        #  rtb=True)