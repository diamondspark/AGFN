from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops

from gflownet.envs.graph_building_env import GraphActionCategorical, GraphActionType

def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes

    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.

    The per node outputs are the concatenation of the final (post graph-convolution) node embeddings
    and of the final virtual node embedding of the graph each node corresponds to.

    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    virtual node embeddings, and of the conditional information embedding.
    """

    def __init__(self, x_dim, e_dim, g_dim, num_emb=64, num_layers=3, num_heads=2, num_noise=0, ln_type="pre"):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features. Equals to the number of features for conditional vector; 
            typically num_thermometer_dim*2*len(conditional_range_dict)
        g_dim: int
            The number of graph-level features. Equals to the number of additional features for conditional vector of finetuning; 
            typically num_thermometer_dim*2*len(ft_conditional_dict)
        num_emb: int
            The number of hidden dimensions, i.e. embedding size. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        ln_type: str
            The location of Layer Norm in the transformer, either 'pre' or 'post', default 'pre'.
            (apparently, before is better than after, see https://arxiv.org/pdf/2002.04745.pdf)
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_noise = num_noise
        assert ln_type in ["pre", "post"]
        self.ln_type = ln_type

        self.x2h = mlp(x_dim + num_noise, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(g_dim, num_emb, num_emb, 2)
        
        # print('gTx.py g dim ft for cond info ',g_dim)
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add", norm=None),
                        gnn.TransformerConv(num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads),
                        nn.Linear(num_heads * num_emb, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                        nn.Linear(num_emb, num_emb * 2),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """Forward pass

        Parameters
        ----------
        g: gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        node_embeddings: torch.Tensor
            Per node embeddings. Shape: (g.num_nodes, self.num_emb).
        graph_embeddings: torch.Tensor
            Per graph embeddings. Shape: (g.num_graphs, self.num_emb * 2).
        """
        # print('Actual gTx.py cond ',cond.shape)
        if self.num_noise > 0:
            x = torch.cat([g.x, torch.rand(g.x.shape[0], self.num_noise, device=g.x.device)], 1)
        else:
            x = g.x
        o = self.x2h(x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat([g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1)
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)
        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)

        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[i * 7 : (i + 1) * 7]
            cs = cscale(c[aug_batch])
            if self.ln_type == "post":
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        glob = torch.cat([gnn.global_mean_pool(o[: -c.shape[0]], g.batch), o[-c.shape[0] :]], 1)
        o_final = torch.cat([o[: -c.shape[0]]], 1)
        return o_final, glob



class GraphTransformerGFN(nn.Module):
    """GraphTransformer class for a GFlowNet which outputs a GraphActionCategorical. Meant for atom-wise
    generation.

    Outputs logits corresponding to the action types used by the env_ctx argument.
    """
    # The GraphTransformer outputs per-node, per-edge, and per-graph embeddings, this routes the
    # embeddings to the right MLP
    _action_type_to_graph_part = {
        GraphActionType.Stop: "graph",
        GraphActionType.AddNode: "node",
        GraphActionType.SetNodeAttr: "node",
        GraphActionType.AddEdge: "non_edge",
        GraphActionType.SetEdgeAttr: "edge",
        GraphActionType.RemoveNode: "node",
        GraphActionType.RemoveNodeAttr: "node",
        GraphActionType.RemoveEdge: "edge",
        GraphActionType.RemoveEdgeAttr: "edge",
    }

    # The torch_geometric batch key each graph part corresponds to
    _graph_part_to_key = {
            "graph": None,
            "node": "x",
            "non_edge": "non_edge_index",
            "edge": "edge_index",
        }

    def __init__(
        self,
        env_ctx,
        num_emb=64,
        num_layers=3,
        num_heads=2,
        num_mlp_layers=0,
        ln_type="pre",
        num_graph_out=1,
        do_bck=False,
        rtb=False
    ):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
            ln_type=ln_type,
        )
        # print('gTx.py env_ctx.num_cond_dim ',env_ctx.num_cond_dim)
        self.env_ctx = env_ctx
        self.num_emb = num_emb
        num_final = num_emb
        self.rtb = rtb
        num_glob_final = num_emb * 2
        num_edge_feat = num_emb if env_ctx.edges_are_unordered else num_emb * 2
        self.edges_are_duplicated = env_ctx.edges_are_duplicated
        self.edges_are_unordered = env_ctx.edges_are_unordered
        self.action_type_order = env_ctx.action_type_order

        # Every action type gets its own MLP that is fed the output of the GraphTransformer.
        # Here we define the number of inputs and outputs of each of those (potential) MLPs.
        self._action_type_to_num_inputs_outputs = {
            GraphActionType.Stop: (num_glob_final, 1),
            GraphActionType.AddNode: (num_final, env_ctx.num_new_node_values),
            GraphActionType.SetNodeAttr: (num_final, env_ctx.num_node_attr_logits),
            GraphActionType.AddEdge: (num_edge_feat, 1),
            GraphActionType.SetEdgeAttr: (num_edge_feat, env_ctx.num_edge_attr_logits),
            GraphActionType.RemoveNode: (num_final, 1),
            GraphActionType.RemoveNodeAttr: (num_final, env_ctx.num_node_attrs - 1),
            GraphActionType.RemoveEdge: (num_edge_feat, 1),
            GraphActionType.RemoveEdgeAttr: (num_edge_feat, env_ctx.num_edge_attrs),
        }
        # print('gtx.py_action_type_to_num_inputs_outputs ', self._action_type_to_num_inputs_outputs)
        self._action_type_to_key = {
            at: self._graph_part_to_key[self._action_type_to_graph_part[at]] for at in self._action_type_to_graph_part
        }

        # Here we create only the embedding -> logit mapping MLPs that are required by the environment
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.do_bck = do_bck
        if do_bck:
            self.bck_action_type_order = env_ctx.bck_action_type_order
        if num_graph_out > 0:
            self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, num_mlp_layers)
        else:
            self.emb2graph_out = None
        # TODO: flag for this
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def _action_type_to_mask(self, t, g):
        return getattr(g, t.mask_name) if hasattr(g, t.mask_name) else torch.ones((1, 1), device=g.x.device)

    def _action_type_to_logit(self, t, emb, g):
        logits = self.mlps[t.cname](emb[self._action_type_to_graph_part[t]])
        return self._mask(logits, self._action_type_to_mask(t, g))

    def _mask(self, x, m):
        # mask logit vector x with binary mask m, -1000 is a tiny log-value
        return x * m + -1000000000 * (1 - m)

    def _make_cat(self, g, emb, action_types):
        # for t in action_types:
        #     print('gtx.py t ',t,' logits ', self._action_type_to_logit(t, emb, g).shape)

        return GraphActionCategorical(
            g,
            logits=[self._action_type_to_logit(t, emb, g) for t in action_types],
            keys=[self._graph_part_to_key[self._action_type_to_graph_part[t]] for t in action_types],
            masks=[self._action_type_to_mask(t, g) for t in action_types],
            types=action_types,
        )

    def forward(self, g: gd.Batch, reverse=False):
        '''reverse: bool 
        If True, return P_B. This is used to compute reverse trajectories for offline mols.
        else, when computing the loss using RTB, do not return P_B to prevent unusd parameters in loss computation error
        '''
        # print('GTgfn forward g', g)
        # print('gtx.py ',next(self.logZ.parameters()).device )
        if g.x.shape[0] == 0:
            return [None]
            node_embeddings, graph_embeddings = (
                torch.zeros((0, self.num_emb), device=g.x.device),
                torch.zeros((0, self.num_emb * 2), device=g.x.device),
            )
        else:
            node_embeddings, graph_embeddings = self.transf(g, g.cond_info)
        # "Non-edges" are edges not currently in the graph that we could add
        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            # If the environment context isn't setting non_edge_index, we can safely assume that
            # action is not in ctx.action_type_order.
            non_edge_embeddings = None
        if self.edges_are_duplicated:
            # On `::2`, edges are typically duplicated to make graphs undirected, only take the even ones
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }
        if self.emb2graph_out is not None:
            graph_out = self.emb2graph_out(graph_embeddings)
        else:
            graph_out = None
        fwd_cat = self._make_cat(g, emb, self.action_type_order)        
        
        # print(f'gtx.py logZ_forward {logZ_forward.shape}, logZ {self.logZ}')
        if self.rtb:
            if reverse:
                bck_cat = self._make_cat(g, emb, self.bck_action_type_order)
                return fwd_cat, bck_cat, graph_out, None #logZ_forward
            else:
                return fwd_cat
            
        logZ_forward = self.logZ(g.cond_info)[:, 0]
        if self.do_bck:
            bck_cat = self._make_cat(g, emb, self.bck_action_type_order)
            return fwd_cat, bck_cat, graph_out, logZ_forward
        
        return fwd_cat, graph_out, logZ_forward


class PrunedGraphTransformerGFN(GraphTransformerGFN):
    '''Removed logZ and  PB part of the network since they do not participate in 
       gradient computation for RTB. Use this class for RTB implementation w/ multigpu DDP train setup
    '''
    def __init__(self, full_model: GraphTransformerGFN, ctx, hps):                                         
        super().__init__(ctx, num_emb=hps["num_emb"], num_layers=hps["num_layers"],
                                         do_bck=False, num_graph_out=int(hps["tb_do_subtb"]), num_mlp_layers=hps['num_mlp_layers'])
                
        # Retain all parts of the model
        self.transf = full_model.transf
        self.mlps = nn.ModuleDict({
            key: full_model.mlps[key]
            for key in full_model.mlps.keys()
            if key not in {
                "remove_edge_attr",
                "remove_edge",
                "remove_node_attr",
                "remove_node"
            }
        })
       
        self.logZ = full_model.logZ
        # Disable gradients for logZ
        for param in self.logZ.parameters():
            param.requires_grad = False
    
    def forward(self, g: gd.Batch):
        if  g.x.shape[0] == 0:
            return [None]
        else:
            node_embeddings, graph_embeddings = self.transf(g, g.cond_info)
        # "Non-edges" are edges not currently in the graph that we could add
        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            # If the environment context isn't setting non_edge_index, we can safely assume that
            # action is not in ctx.action_type_order.
            non_edge_embeddings = None
        if self.edges_are_duplicated:
            # On `::2`, edges are typically duplicated to make graphs undirected, only take the even ones
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }
        if self.emb2graph_out is not None:
            graph_out = self.emb2graph_out(graph_embeddings)
        else:
            graph_out = None
        fwd_cat = self._make_cat(g, emb, self.action_type_order)        
        logZ_forward = self.logZ(g.cond_info)[:, 0]
        
        return fwd_cat, logZ_forward

class PrunedGraphTransformerGFNPrior(GraphTransformerGFN):
    def __init__(self, full_model: GraphTransformerGFN, ctx, hps):                                         
        super().__init__(ctx, num_emb=hps["num_emb"], num_layers=hps["num_layers"],
                                         do_bck=True, num_graph_out=int(hps["tb_do_subtb"]), num_mlp_layers=hps['num_mlp_layers'])#, rtb =True)
                
        # Retain all parts of the model
        self.transf = full_model.transf
        self.mlps = full_model.mlps
        self.logZ = None

    
    def forward(self, g: gd.Batch, reverse=False):
        '''reverse: bool 
        If True, return P_B. This is used to compute reverse trajectories for offline mols.
        else, when computing the loss using RTB, do not return P_B to prevent unusd parameters in loss computation error
        '''

        if g.x.shape[0] == 0:
            return [None]
        else:
            node_embeddings, graph_embeddings = self.transf(g, g.cond_info)
        # "Non-edges" are edges not currently in the graph that we could add
        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            # If the environment context isn't setting non_edge_index, we can safely assume that
            # action is not in ctx.action_type_order.
            non_edge_embeddings = None
        if self.edges_are_duplicated:
            # On `::2`, edges are typically duplicated to make graphs undirected, only take the even ones
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }
        if self.emb2graph_out is not None:
            graph_out = self.emb2graph_out(graph_embeddings)
        else:
            graph_out = None
        fwd_cat = self._make_cat(g, emb, self.action_type_order)        
        
        # if self.rtb:
        if reverse:
            bck_cat = self._make_cat(g, emb, self.bck_action_type_order)
            return fwd_cat, bck_cat, graph_out, None #logZ_forward
        else:
            return fwd_cat


