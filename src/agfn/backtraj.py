from gflownet.envs.graph_building_env import GraphActionType, Graph, GraphAction
import torch
from rdkit import Chem

class Reverse():
    def __init__(self, pretrainer):
        self.env = pretrainer.env
        self.task = pretrainer.task
        self.ctx = pretrainer.ctx
        self.hps = pretrainer.hps
        self.rng = pretrainer.rng
    
    def reverse_action(self, g: Graph, ga: GraphAction):

        ##Reverse forward actions
        if ga.action == GraphActionType.Stop:
            return ga
        if ga.action == GraphActionType.AddNode:
            return GraphAction(GraphActionType.RemoveNode, source=len(g.nodes))
        if ga.action == GraphActionType.AddEdge:
            return GraphAction(GraphActionType.RemoveEdge, source=ga.source, target=ga.target)
        if ga.action == GraphActionType.SetNodeAttr:
            return GraphAction(GraphActionType.RemoveNodeAttr, source=ga.source, attr=ga.attr)
        if ga.action == GraphActionType.SetEdgeAttr:
            return GraphAction(GraphActionType.RemoveEdgeAttr, source=ga.source, target=ga.target, attr=ga.attr)
        
        ##Reverse backward actions
        if ga.action == GraphActionType.RemoveNode:
            neighbors = [i for i in g.edges if i[0]==ga.source or i[1]==ga.source]
            assert len(neighbors) <= 1  # RemoveNode should only be a legal action if the node has one or zero neighbors
            source = 0 if not len(neighbors) else neighbors[0][0] if neighbors[0][0]!=ga.source else neighbors[0][1]
            return GraphAction(GraphActionType.AddNode, source=source, value = g.nodes[ga.source]["v"])
        if ga.action == GraphActionType.RemoveEdge:
            return GraphAction(GraphActionType.AddEdge,source=ga.source, target=ga.target)
        if ga.action == GraphActionType.RemoveNodeAttr:
            return GraphAction(GraphActionType.SetNodeAttr, source=ga.source, target=ga.target, attr = ga.attr,
                            value = g.nodes[ga.source][ga.attr])
        if ga.action == GraphActionType.RemoveEdgeAttr:
            return GraphAction(GraphActionType.SetEdgeAttr, source=ga.source, target=ga.target, attr = ga.attr, 
                            value= g.edges[ga.source, ga.target][ga.attr])

    def relabel(self, g: Graph, ga: GraphAction):
        """Relabel the nodes for g to 0-N, and the graph action ga applied to g.
        This is necessary because torch_geometric and EnvironmentContext classes expect nodes to be
        labeled 0-N, whereas GraphBuildingEnv.parent can return parents with e.g. a removed node that
        creates a gap in 0-N, leading to a faulty encoding of the graph.
        """
        rmap = dict(zip(g.nodes, range(len(g.nodes))))
        if not len(g) and ga.action == GraphActionType.AddNode:
            rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
        g = g.relabel_nodes(rmap)
        if ga.source is not None:
            ga.source = rmap[ga.source]
        if ga.target is not None:
            ga.target = rmap[ga.target]
        return ga, g

    def reverse_trajectory(self, batch, cond_info_bck, model, device): #getdata()
        '''
        Returns data batch wiht backward trajectory sampled from the P_B model 
        params: 
        batch: [SMILES]
        ctx: Molecular Context object
        env : GraphBuildingEnvironment
        task: SEHFragTask
        '''
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]
        
        batch_size = len(batch)
        mols = [Chem.MolFromSmiles(smiles) for smiles in batch]
        graphs = [self.ctx.mol_to_graph(m) for m in mols]
        done = [False]*len(batch)
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(len(batch))]
        bck_logprob = [[] for i in range(batch_size)]
        fwd_a = [[] for i in range(batch_size)]
        # t =0 for all the not done graphs in the batch
        # t =1 for all the not done graphs in the batch
        # ....
        # 
        cond_info_bck = cond_info_bck.to(device)
        while sum(done)<len(batch): #all graphs is the batch are not done i.e. until done==[True]*n
            # print('getdata() trying to make torch graphs ')
            torch_graphs = []
            for i in not_done(range(batch_size)):
                try:
                    graph = graphs[i]
                    cond_info_current = cond_info_bck[i][None, :].cpu()
                    torch_graph = self.ctx.graph_to_Data(graph, cond_info=cond_info_current)
                    torch_graphs.append(torch_graph)
                except Exception as e:
                    print('bcktraj.py e', e, i, len(batch), graph)
                    print('backtraj.py batch[i]', batch.iloc[i])


            torch_graphs = [
                self.ctx.graph_to_Data(graphs[i], cond_info=cond_info_bck[i][None, :].cpu()) for i in not_done(range(batch_size))
            ]
            # not_done_mask = torch.tensor(done, device=device).logical_not()
            # with torch.no_grad():
            fwd_cat, bck_cat, log_reward_preds, _ = model(self.ctx.collate(torch_graphs).to(device))
            #TODO: if random_action_prob>
            if self.hps['random_action_prob']>0:
                masks = [1] * len(bck_cat.logits) if bck_cat.masks is None else bck_cat.masks
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < self.hps['random_action_prob'], device=device
                ).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                bck_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(bck_cat.logits, masks, bck_cat.batch)
                ]
            bck_actions = bck_cat.sample()

            graph_bck_actions = [self.ctx.aidx_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)]

            bck_log_probs = bck_cat.log_prob(bck_actions)
            for i, j in zip(not_done(range(batch_size)), range(batch_size)):# for each of the not done trajectory (mols in graphs list), apply the respective sampled action
                if not done[i]:
                    bck_logprob[i].append(bck_log_probs[j].unsqueeze(0))
                    data[i]["traj"].append((graphs[i],graph_bck_actions[j]))
                    
                    reverse_action = self.reverse_action(graphs[i],graph_bck_actions[j])
                    
                    data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1).detach()#.cpu()
                    gp_temp = self.env.step(graphs[i],graph_bck_actions[j])
                    try:
                        relabeled_rev_act, graphs[i] = self.relabel(gp_temp, reverse_action)
                        # print(relabeled_rev_act)
                        # print('graphs' , graphs[i])
                    except Exception as e:
                        print('bcktraj.py exception', e)
                        # print(data[i])
                        print('graphs[i] ' ,i,  graphs[i])
                        print('smiles[i] ',i, Chem.MolToSmiles(mols[i]))
                    
                    fwd_a[i].append(relabeled_rev_act)
                    data[i]["fwd_a"] = fwd_a[i]
                    
                    
                    if len(graphs[i])==0: done[i] = True 

        return data
        

    def flip_trajectory(self,data):
        '''
        Given a backwward trajectory computed by sampling bck_actions from P_B,
        this method flips it to a forward trajectory such that the resulting 
        trajectory is compatible with GFN algo (Trajectory balance) computation code.
        param:
        data: batch dictionary containing backward trajectory
        '''
        flipped_data = []
        for i,mol in enumerate(data):
            my_dict = {'traj':[],'reward_pred':None, 'is_valid':True, 'is_sink':[0]*(len(mol['traj'])), 
                    'bck_a':[], 'bck_logprobs': torch.flip(mol['bck_logprobs'], dims = (0,))}
            flippy_traj, bck_a = [],[GraphAction(GraphActionType.Stop)]
        
            for j in range(len(mol['traj'])):
                bck_a.append(mol['traj'][-(j+1)][1])
                if j ==0:
                    flippy_traj.append((self.env.new(), mol['fwd_a'][-(j+1)]))
                else:
                    flippy_traj.append(( mol['traj'][-j][0] , mol['fwd_a'][-(j+1)]))
            
            flippy_traj.append((mol['traj'][0][0], GraphAction(GraphActionType.Stop)))
            my_dict['traj'] = flippy_traj
            my_dict['bck_a'] = bck_a
            my_dict['is_sink'].append(1)
            flipped_data.append( my_dict)
        
        return flipped_data

class ReverseFineTune():
    def __init__(self, env,ctx,hps,rng):
        self.env = env
        # self.task = task
        self.ctx = ctx
        # self.rev_cond_info = pretrainer.cond_info['encoding'][-pretrainer.num_offline:]#.to(pretrainer.device)
        self.hps = hps
        self.rng = rng
    
    def reverse_action(self, g: Graph, ga: GraphAction):

        ##Reverse forward actions
        if ga.action == GraphActionType.Stop:
            return ga
        if ga.action == GraphActionType.AddNode:
            return GraphAction(GraphActionType.RemoveNode, source=len(g.nodes))
        if ga.action == GraphActionType.AddEdge:
            return GraphAction(GraphActionType.RemoveEdge, source=ga.source, target=ga.target)
        if ga.action == GraphActionType.SetNodeAttr:
            return GraphAction(GraphActionType.RemoveNodeAttr, source=ga.source, attr=ga.attr)
        if ga.action == GraphActionType.SetEdgeAttr:
            return GraphAction(GraphActionType.RemoveEdgeAttr, source=ga.source, target=ga.target, attr=ga.attr)
        
        ##Reverse backward actions
        if ga.action == GraphActionType.RemoveNode:
            # TODO: implement neighbors or something
            # neighbors = list(g.neighbors(ga.source))
            #source = 0 if not len(neighbors) else neighbors[0]
            neighbors = [i for i in g.edges if i[0]==ga.source or i[1]==ga.source]
            assert len(neighbors) <= 1  # RemoveNode should only be a legal action if the node has one or zero neighbors
            source = 0 if not len(neighbors) else neighbors[0][0] if neighbors[0][0]!=ga.source else neighbors[0][1]
            return GraphAction(GraphActionType.AddNode, source=source, value = g.nodes[ga.source]["v"])
        if ga.action == GraphActionType.RemoveEdge:
            return GraphAction(GraphActionType.AddEdge,source=ga.source, target=ga.target)
        if ga.action == GraphActionType.RemoveNodeAttr:
            return GraphAction(GraphActionType.SetNodeAttr, source=ga.source, target=ga.target, attr = ga.attr,
                            value = g.nodes[ga.source][ga.attr])
        if ga.action == GraphActionType.RemoveEdgeAttr:
            return GraphAction(GraphActionType.SetEdgeAttr, source=ga.source, target=ga.target, attr = ga.attr, 
                            value= g.edges[ga.source, ga.target][ga.attr])

    def relabel(self, g: Graph, ga: GraphAction):
        """Relabel the nodes for g to 0-N, and the graph action ga applied to g.
        This is necessary because torch_geometric and EnvironmentContext classes expect nodes to be
        labeled 0-N, whereas GraphBuildingEnv.parent can return parents with e.g. a removed node that
        creates a gap in 0-N, leading to a faulty encoding of the graph.
        """
        rmap = dict(zip(g.nodes, range(len(g.nodes))))
        if not len(g) and ga.action == GraphActionType.AddNode:
            rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
        g = g.relabel_nodes(rmap)
        if ga.source is not None:
            try:
                ga.source = rmap[ga.source]  #TODO: Better error handling
            except Exception as e:
                print('bcktraj.py g',Chem.MolToSmiles(self.ctx.graph_to_mol(g)))
        if ga.target is not None:
            ga.target = rmap[ga.target]
        return ga, g

    def reverse_trajectory(self, batch, cond_info_bck, model, device): #getdata()
        '''
        Returns data batch wiht backward trajectory sampled from the P_B model 
        params: 
        batch: [SMILES]
        ctx: Molecular Context object
        env : GraphBuildingEnvironment
        task: SEHFragTask
        '''
      
        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]
        
        batch_size = len(batch)
        mols = [Chem.MolFromSmiles(smiles) for smiles in batch]
        graphs = [self.ctx.mol_to_graph(m) for m in mols]
        done = [False]*len(batch)
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(len(batch))]
        bck_logprob = [[] for i in range(batch_size)]
        fwd_a = [[] for i in range(batch_size)]
        # t =0 for all the not done graphs in the batch
        # t =1 for all the not done graphs in the batch
        # ....
        # 
        cond_info_bck = cond_info_bck.to(device)
        while sum(done)<len(batch): #all graphs is the batch are not done i.e. until done==[True]*n
            torch_graphs = [
                self.ctx.graph_to_Data(graphs[i], cond_info=cond_info_bck[i][None, :].cpu()) for i in not_done(range(batch_size))
            ]

            fwd_cat, bck_cat, log_reward_preds, _ = model(self.ctx.collate(torch_graphs).to(device), reverse=True)

            #TODO: if random_action_prob>
            if self.hps['random_action_prob']>0:
                masks = [1] * len(bck_cat.logits) if bck_cat.masks is None else bck_cat.masks
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    self.rng.uniform(size=len(torch_graphs)) < self.hps['random_action_prob'], device=device
                ).float()
                # Set the logits to some large value if they're not masked, this way the masked
                # actions have no probability of getting sampled, and there is a uniform
                # distribution over the rest
                bck_cat.logits = [
                    # We don't multiply m by i on the right because we're assume the model forward()
                    # method already does that
                    is_random_action[b][:, None] * torch.ones_like(i) * m * 100 + i * (1 - is_random_action[b][:, None])
                    for i, m, b in zip(bck_cat.logits, masks, bck_cat.batch)
                ]
            if False: #self.hps['sample_temp']!=1: #Q: what is sample_temp
                sample_bck_cat = copy.copy(bck_cat)
                sample_bck_cat.logits = [i / hps['sample_temp'] for i in bck_cat.logits]
                bck_actions = sample_cat.sample()
            else:
                bck_actions = bck_cat.sample()

            graph_bck_actions = [self.ctx.aidx_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)]

            bck_log_probs = bck_cat.log_prob(bck_actions)
            for i, j in zip(not_done(range(batch_size)), range(batch_size)):# for each of the not done trajectory (mols in graphs list), apply the respective sampled action
                if not done[i]:
                    bck_logprob[i].append(bck_log_probs[j].unsqueeze(0))
                    data[i]["traj"].append((graphs[i],graph_bck_actions[j]))
                    
                    reverse_action = self.reverse_action(graphs[i],graph_bck_actions[j])
                    
                    data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1).detach()#.cpu()
                    gp_temp = self.env.step(graphs[i],graph_bck_actions[j])
                    relabeled_rev_act, graphs[i] = self.relabel(gp_temp, reverse_action)
                    
                    fwd_a[i].append(relabeled_rev_act)
                    data[i]["fwd_a"] = fwd_a[i]
                    
                    
                    if len(graphs[i])==0: done[i] = True 

        return data
        

    def flip_trajectory(self,data):
        '''
        Given a backwward trajectory computed by sampling bck_actions from P_B,
        this method flips it to a forward trajectory such that the resulting 
        trajectory is compatible with GFN algo (Trajectory balance) computation code.
        param:
        data: batch dictionary containing backward trajectory
        '''
        flipped_data = []
        for i,mol in enumerate(data):
            my_dict = {'traj':[],'reward_pred':None, 'is_valid':True, 'is_sink':[0]*(len(mol['traj'])), 
                    'bck_a':[], 'bck_logprobs': torch.flip(mol['bck_logprobs'], dims = (0,))}
            flippy_traj, bck_a = [],[GraphAction(GraphActionType.Stop)]
        
            for j in range(len(mol['traj'])):
                bck_a.append(mol['traj'][-(j+1)][1])
                if j ==0:
                    flippy_traj.append((self.env.new(), mol['fwd_a'][-(j+1)]))
                else:
                    flippy_traj.append(( mol['traj'][-j][0] , mol['fwd_a'][-(j+1)]))
            
            flippy_traj.append((mol['traj'][0][0], GraphAction(GraphActionType.Stop)))
            my_dict['traj'] = flippy_traj
            my_dict['bck_a'] = bck_a
            my_dict['is_sink'].append(1)
            flipped_data.append( my_dict)
        
        return flipped_data

