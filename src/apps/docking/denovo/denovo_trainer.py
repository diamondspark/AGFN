from reward_dock import RewardDockFineTune
from finetuneSrc.finetuner import FineTunerRTB

class DockingFineTuner(FineTunerRTB):
    def __init__(self, hps, conditional_range_dict, cond_prop_var, load_path, rank, world_size, gfn_samples_path, ft_conditionals_dict=None):
        super().__init__(hps, conditional_range_dict, cond_prop_var, load_path, rank, world_size, ft_conditionals_dict)
        self.reward = RewardDockFineTune(conditional_range_dict,ft_conditionals_dict, cond_prop_var, hps["reward_aggergation"], hps['atomenv_dictionary'], hps['zinc_rad_scale'], hps, gfn_samples_path)   
        self.gfn_samples_path = gfn_samples_path