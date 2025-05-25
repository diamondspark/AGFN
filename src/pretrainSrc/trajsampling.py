from agfn.backtraj import Reverse
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

class Trajectory_Sampling():
    """
    Wrapper class to sample forward and backward actions 
    from P_F and P_B respectively, and thus form forward
    and backward trajectories.
    """
    def __init__(self,pretrainer):
        self.reward = pretrainer.reward
        self.task = pretrainer.task
        self.model = pretrainer.gfn_trainer.model
        self.graph_sampler = pretrainer.graph_sampler
        self.reverse = Reverse(pretrainer)
        self.hps = pretrainer.hps
        if 'seed_smiles' in self.hps:
            if self.hps.get('seed_scaffold',None):
                print('trajsamp.py Optimizing ', self.hps['seed_scaffold'], ' scaffold provided')
                scaffold = Chem.MolFromSmiles(self.hps['seed_scaffold'])
            else:
                print('trajsamp.py Optimizing ', self.hps['seed_smiles'])
                mol = Chem.MolFromSmiles(self.hps['seed_smiles'])
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            self.seed_graph = pretrainer.ctx.mol_to_graph(scaffold)
            print('trajsamp.py seed_graph ', self.seed_graph)
        else:
            self.seed_graph = None
        print('seed graph is ', self.seed_graph)

    def get_tau_forward_from_model(self,model,dev,n_trajs,cond_info):
        device = dev#self.pretrainer.device
        num_samples = n_trajs
        # from trajectory balance (sampling trajectory for model (gives forward trajectory and also
        # back action -> can form back trajectories from these back actions (onpolicy backward trajectories) 
        # if needed))
        # result = self.pretrainer.graph_sampler.sample_from_model(model,num_samples,
        #                                                          cond_info.to(device),
        #                                                          device)
        result = self.graph_sampler.sample_from_model(model,num_samples,
                                                    cond_info.to(device),
                                                    device,
                                                    random_stop_action_prob= self.hps['random_stop_prob'],
                                                    random_action_prob = self.hps['random_action_prob'],
                                                    seed_graph= self.seed_graph
                                                    )
        trajs = [r['traj'] for r in result]
        return trajs, result


    def sample_tau_from_PB(self, smiles_batch, model, device):
        # t0 = time.time()
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_batch if smiles is not None]
        lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew = self.reward.molecular_rewards(mols)
        cond_info_bck = self.task.compute_cond_info_backward(flat_rewards)
        self.cond_info_bck_encoding = self.task.thermometer_encoding(cond_info_bck)
        data = self.reverse.reverse_trajectory(smiles_batch,self.cond_info_bck_encoding, model, device)
        # print(f'getdata() took {time.time()-t0} s')
        flipped_data = self.reverse.flip_trajectory(data)
        return data, flipped_data, (lg_rewards, flat_rewards, total_reward, zinc_flat_offln_rew)
    
    def sample_tau_from_PF(self,model,dev,n_trajs, ft_conditionals_dict=None):
        # cond_info =self.pretrainer.cond_info["encoding"][:self.pretrainer.num_online]
        cond_info = self.task.compute_cond_info_forward(n_trajs)
        self.cond_info_encoding = self.task.thermometer_encoding(cond_info)
        # cond_info_ft = self.task.compute_cond_info_forward(n_trajs, ft_conditionals_dict)
        # self.cond_info_ft_encoding = self.task.thermometer_encoding(cond_info_ft)

        trajs, result = self.get_tau_forward_from_model(model,dev, n_trajs,self.cond_info_encoding)#, self.cond_info_ft_encoding)
        return trajs, result
