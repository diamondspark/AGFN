import copy
from typing import List, Tuple
import warnings

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import torch
import torch_geometric.data as gd
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import BondType, ChiralType

from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnvContext,
    graph_without_edge,
)
from gflownet.utils.graphs import random_walk_probs
try:
    from gflownet._C import mol_graph_to_Data, Graph as C_Graph, GraphDef, Data_collate
    print('molbuildingenv.py C is available; but not using it')
    C_Graph_available = not True  
except ImportError:
    print('molbuildingenv.py c import error')
    warnings.warn("Could not import mol_graph_to_Data, Graph, GraphDef from _C, using pure python implementation")
    C_Graph_available = False

DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]

class BatchList:
    def __init__(self, l):
        self._list = l

    def append(self, item):
        return self._list.append(item)

    def __getitem__(self, index):
        return self._list.__getitem__(index)

    def __len__(self):
        return self._list.__len__()

    def to(self, devs):
        if isinstance(devs, torch.device):
            devs = [devs] * len(self)
        self._list = [i.to(dev) for i, dev in zip(self, devs)]
        for i in self.__dict__:
            if isinstance(self.__dict__[i], torch.Tensor):
                self.__dict__[i] = self.__dict__[i].to(devs[0])
        return self

class MolBuildingEnvContext(GraphBuildingEnvContext):
    """A specification of what is being generated for a GraphBuildingEnv

    This context specifies how to create molecules atom-by-atom (and attribute-by-attribute).
    """

    def __init__(
        self,
        atoms=["C", "N", "O", "F", "P", "S"],
        num_cond_dim=0,
        num_ft_cond_dim=0,
        chiral_types=DEFAULT_CHIRAL_TYPES,
        charges=[0, 1, -1],
        expl_H_range=[0, 1],
        allow_explicitly_aromatic=False,
        num_rw_feat=8,
        max_nodes=None,
        max_edges=None,
    ):
        """An env context for building molecules atom-by-atom and bond-by-bond.

        Parameters
        ----------
        atoms: List[str]
            The list of allowed atoms. (default, [C, N, O, F, P, S], the six "biological" elements)
        num_cond_dim: int
            The number of dimensions the conditioning vector will have. (default 0)
        num_cond_dim: int
            The number of dimensions of the fine tuning conditioning vector will have. (default 0)
        chiral_types: List[rdkit.Chem.rdchem.ChiralType]
            The list of allowed chiral types. (default [unspecified, CW, CCW])
        charges: List[int]
            The list of allowed charges on atoms. (default [0, 1, -1])
        expl_H_range: List[int]
            The list of allowed explicit # of H values. (default [0, 1])
        allow_explicitly_aromatic: bool
            If true, then the agent is allowed to set bonds to be aromatic, otherwise the agent has to
            generate a Kekulized version of aromatic rings and we rely on rdkit to recover aromaticity.
            (default False)
        num_rw_feat: int
            If >0, augments the feature representation with n-step random walk features. (default n=8).
        max_nodes: int
            If not None, then the maximum number of nodes in the graph. Corresponding actions are masked. (default None)
        max_edges: int
            If not None, then the maximum number of edges in the graph. Corresponding actions are masked. (default None)
        """
        # idx 0 has to coincide with the default value
        self.atom_attr_values = {
            "v": atoms, # + ["*"],
            "chi": chiral_types,  #uncomment if using chirality
            "charge": charges,
            "expl_H": expl_H_range,
            "no_impl": [False, True],
            # "fill_wildcard": [None] + atoms,  # default is, there is nothing
        }
        self.num_rw_feat = num_rw_feat
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.collate_split = 0
        self.charges = charges
        self.default_wildcard_replacement = "C"
        self.negative_attrs = ["fill_wildcard"]
        self.atom_attr_defaults = {k: self.atom_attr_values[k][0] for k in self.atom_attr_values}
        # The size of the input vector for each atom
        self.atom_attr_size = sum(len(i) for i in self.atom_attr_values.values())
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        # 'v' is set separately when creating the node, so there's no point in having a SetNodeAttr logit for it
        self.settable_atom_attrs = [i for i in self.atom_attrs if i != "v"]
        # The beginning position within the input vector of each attribute
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        # The beginning position within the logit vector of each attribute
        num_atom_logits = [len(self.atom_attr_values[i]) - 1 for i in self.settable_atom_attrs]
        self.atom_attr_logit_slice = {
            k: (s, e)
            for k, s, e in zip(
                self.settable_atom_attrs, [0] + list(np.cumsum(num_atom_logits)), np.cumsum(num_atom_logits)
            )
        }
        # The attribute and value each logit dimension maps back to
        self.atom_attr_logit_map = [
            (k, v)
            for k in self.settable_atom_attrs
            # index 0 is skipped because it is the default value
            for v in self.atom_attr_values[k][1:]
        ]

        # By default, instead of allowing/generating aromatic bonds, we instead "ask of" the
        # generative process to generate a Kekulized form of the molecule. RDKit is capable of
        # recovering aromatic ring, and so we leave it at that.
        self.allow_explicitly_aromatic = allow_explicitly_aromatic
        aromatic_optional = [BondType.AROMATIC] if allow_explicitly_aromatic else []
        self.bond_attr_values = {
            "type": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE] + aromatic_optional,
        }
        self.bond_attr_defaults = {k: self.bond_attr_values[k][0] for k in self.bond_attr_values}
        self.bond_attr_size = sum(len(i) for i in self.bond_attr_values.values())
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        num_bond_logits = [len(self.bond_attr_values[i]) - 1 for i in self.bond_attrs]
        self.bond_attr_logit_slice = {
            k: (s, e)
            for k, s, e in zip(self.bond_attrs, [0] + list(np.cumsum(num_bond_logits)), np.cumsum(num_bond_logits))
        }
        self.bond_attr_logit_map = [(k, v) for k in self.bond_attrs for v in self.bond_attr_values[k][1:]]
        self._bond_valence = {
            BondType.SINGLE: 1,
            BondType.DOUBLE: 2,
            BondType.TRIPLE: 3,
            BondType.AROMATIC: 1.5,
        }
        pt = Chem.GetPeriodicTable()
        self._max_atom_valence = {
            **{a: max(pt.GetValenceList(a)) for a in atoms},
            "N": 3,  # We'll handle nitrogen valence later explicitly in graph_to_Data
            "*": 0,  # wildcard atoms have 0 valence until filled in
        }

        # These values are used by Models to know how many inputs/logits to produce
        self.num_new_node_values = len(atoms)
        self.num_node_attr_logits = len(self.atom_attr_logit_map)
        self.num_node_dim = self.atom_attr_size + 1 + self.num_rw_feat
        self.num_edge_attr_logits = len(self.bond_attr_logit_map)
        self.num_edge_dim = self.bond_attr_size
        self.num_node_attrs = len(self.atom_attrs)
        self.num_edge_attrs = len(self.bond_attrs)
        self.num_cond_dim = num_cond_dim
        self.num_ft_cond_dim = num_ft_cond_dim
        self.edges_are_duplicated = True
        self.edges_are_unordered = True

        # Order in which models have to output logits
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.SetNodeAttr,
            GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr,
        ]
        self.bck_action_type_order = [
            GraphActionType.RemoveNode,
            GraphActionType.RemoveNodeAttr,
            GraphActionType.RemoveEdge,
            GraphActionType.RemoveEdgeAttr,
        ]
        self.device = torch.device("cpu")
        if C_Graph_available:
            self.graph_def = GraphDef(self.atom_attr_values, self.bond_attr_values)
            self.graph_cls = self._make_C_graph
        else:
            self.graph_cls = Graph

    def _make_C_graph(self):
        return C_Graph(self.graph_def)

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int], fwd: bool = True):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction"""
        act_type, act_row, act_col = [int(i) for i in action_idx]
        if fwd:
            t = self.action_type_order[act_type]
        else:
            t = self.bck_action_type_order[act_type]
        # print('molbuildingenv.py sampled actiontuple ', action_idx,' actiontype ',act_type, ' t ',t)
        if self.graph_cls is not Graph:
            return g.mol_aidx_to_GraphAction((act_type, act_row, act_col), t)
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=self.atom_attr_values["v"][act_col])
        elif t is GraphActionType.SetNodeAttr:
            attr, val = self.atom_attr_logit_map[act_col]
            return GraphAction(t, source=act_row, attr=attr, value=val)
        elif t is GraphActionType.AddEdge:
            a, b = g.non_edge_index[:, act_row]
            return GraphAction(t, source=a.item(), target=b.item())
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            attr, val = self.bond_attr_logit_map[act_col]
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)
        elif t is GraphActionType.RemoveNode:
            return GraphAction(t, source=act_row)
        elif t is GraphActionType.RemoveNodeAttr:
            attr = self.settable_atom_attrs[act_col]
            return GraphAction(t, source=act_row, attr=attr)
        elif t is GraphActionType.RemoveEdge:
            a, b = g.edge_index[:, act_row * 2]
            return GraphAction(t, source=a.item(), target=b.item())
        elif t is GraphActionType.RemoveEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]
            attr = self.bond_attrs[act_col]
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr)

    def GraphAction_to_aidx(self, g: gd.Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple"""
        for u in [self.action_type_order, self.bck_action_type_order]:
            if action.action in u:
                type_idx = u.index(action.action)
                break
        else:
            raise ValueError(f"Unknown action type {action.action}")
        if self.graph_cls is not Graph:
            return (type_idx,) + g.mol_GraphAction_to_aidx(action)

        if action.action is GraphActionType.Stop:
            row = col = 0
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = self.atom_attr_values["v"].index(action.value)
        elif action.action is GraphActionType.SetNodeAttr:
            row = action.source
            # - 1 because the default is index 0
            col = (
                self.atom_attr_values[action.attr].index(action.value) - 1 + self.atom_attr_logit_slice[action.attr][0]
            )
        elif action.action is GraphActionType.AddEdge:
            # Here we have to retrieve the index in non_edge_index of an edge (s,t)
            # that's also possibly in the reverse order (t,s).
            # That's definitely not too efficient, can we do better?
            row = (
                (g.non_edge_index.T == torch.tensor([(action.source, action.target)])).prod(1)
                + (g.non_edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)
            ).argmax()
            col = 0
        elif action.action is GraphActionType.SetEdgeAttr:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            # row = ((g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1) +
            #       (g.edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)).argmax()
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode="floor")  # type: ignore
            col = (
                self.bond_attr_values[action.attr].index(action.value) - 1 + self.bond_attr_logit_slice[action.attr][0]
            )
        elif action.action is GraphActionType.RemoveNode:
            row = action.source
            col = 0
        elif action.action is GraphActionType.RemoveNodeAttr:
            row = action.source
            col = self.settable_atom_attrs.index(action.attr)
        elif action.action is GraphActionType.RemoveEdge:
            row = ((g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1)).argmax()
            row = int(row) // 2  # edges are duplicated, but edge logits are not
            col = 0
        elif action.action is GraphActionType.RemoveEdgeAttr:
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            row = row.div(2, rounding_mode="floor")  # type: ignore
            col = self.bond_attrs.index(action.attr)
        else:
            raise ValueError(f"Unknown action type {action.action}")
        return (type_idx, int(row), int(col))

    def graph_to_Data(self, g: Graph, cond_info: torch.Tensor, ft_cond_info: torch.Tensor = None) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        if hasattr(g, "_cached_Data"):
            return g._cached_Data
        if C_Graph_available:
            data = mol_graph_to_Data(g, self, torch, cond_info) 
            # data = mol_graph_to_Data(g, self, torch, cond_info, ft_cond_info) # TODO: update mol_graph_to_Data() to use ft_cond_info
            #data = gd.Data(**{k: v for k, v in data.items()}, cond_info=cond_info)
            #g._cached_Data = data
            #return data
            C_data = data
            if 1:
                return data
            gp = Graph()
            nkeys = self.atom_attr_values.keys()
            for i in g.nodes:
                gp.add_node(i, v=g.nodes[i]['v'])
                for k in nkeys:
                    if k == 'v': continue
                    if k in g.nodes[i]:
                        gp.nodes[i][k] = g.nodes[i][k]
            ekeys = self.bond_attr_values.keys()
            for i in g.edges:
                gp.add_edge(*i)
                for k in ekeys:
                    if k in g.edges[i]:
                        gp.edges[i][k] = g.edges[i][k]
            g = gp
        zeros = lambda s: np.zeros(s, dtype=np.float32)
        ones = lambda s: np.ones(s, dtype=np.float32)
        x = zeros((max(1, len(g.nodes)), self.num_node_dim - self.num_rw_feat))
        x[0, -1] = len(g.nodes) == 0
        add_node_mask = ones((x.shape[0], self.num_new_node_values))
        if self.max_nodes is not None and len(g.nodes) >= self.max_nodes:
            add_node_mask *= 0
        remove_node_mask = zeros((x.shape[0], 1)) + (1 if len(g) == 0 else 0)
        remove_node_attr_mask = zeros((x.shape[0], len(self.settable_atom_attrs)))

        explicit_valence = {}
        max_valence = {}
        set_node_attr_mask = ones((x.shape[0], self.num_node_attr_logits))
        # for all atoms, if editable is false, make set_node_attr_mask as false
        bridges = set(nx.bridges(g))
        if not len(g.nodes):
            set_node_attr_mask *= 0
        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            if g.degree(n) <= 1 and len(ad) == 1 and all([len(g[n][neigh]) == 0 for neigh in g.neighbors(n)]):
                # If there's only the 'v' key left and the node is a leaf, and the edge that connect to the node have
                # no attributes set, we can remove it
                remove_node_mask[i] = 1
            for k, sl in zip(self.atom_attrs, self.atom_attr_slice):
                # idx > 0 means that the attribute is not the default value
                # try:
                idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                # except Exception as e:
                    # print('molbuildingenv.py atom)attrvalues ', self.atom_attr_values, ' k ',k, ' ad ', ad)
                x[i, sl + idx] = 1
                if k == "v":
                    continue
                # If the attribute
                #   - is already there (idx > 0),
                #   - or the attribute is a negative attribute and has been filled
                #   - or the attribute is a negative attribute and is not fillable (i.e. not a key of ad)
                # then mask forward logits.
                # For backward logits, positively mask if the attribute is there (idx > 0).
                if k in self.negative_attrs:
                    if k in ad and idx > 0 or k not in ad:
                        s, e = self.atom_attr_logit_slice[k]
                        set_node_attr_mask[i, s:e] = 0
                        # We don't want to make the attribute removable if it's not fillable (i.e. not a key of ad)
                        if k in ad:
                            remove_node_attr_mask[i, self.settable_atom_attrs.index(k)] = 1
                elif k in ad:
                    s, e = self.atom_attr_logit_slice[k]
                    set_node_attr_mask[i, s:e] = 0
                    remove_node_attr_mask[i, self.settable_atom_attrs.index(k)] = 1
            # Account for charge and explicit Hs in atom as limiting the total valence
            max_atom_valence = self._max_atom_valence[ad.get("fill_wildcard", None) or ad["v"]]
            # Special rule for Nitrogen
            if ad["v"] == "N" and ad.get("charge", 0) == 1:
                # This is definitely a heuristic, but to keep things simple we'll limit Nitrogen's valence to 3 (as
                # per self._max_atom_valence) unless it is charged, then we make it 5.
                # This keeps RDKit happy (and is probably a good idea anyway).
                max_atom_valence = 5
            max_valence[n] = max_atom_valence - abs(ad.get("charge", 0)) - ad.get("expl_H", 0)
            # Compute explicitly defined valence:
            explicit_valence[n] = 0
            for ne in g[n]:
                explicit_valence[n] += self._bond_valence[g.edges[(n, ne)].get("type", self.bond_attr_defaults["type"])]
            # If the valence is maxed out, mask out logits that would add a new atom + single bond to this node
            if explicit_valence[n] >= max_valence[n]:
                add_node_mask[i, :] = 0
                if ad["v"] == "N" and ad.get("charge", 0) == 1:
                    # Special case: if the node is a positively charged Nitrogen, and the valence is maxed out (i.e. 5)
                    # the agent cannot remove the charge, it has to remove the bonds (or bond attrs) making this valence
                    # maxed out first.
                    remove_node_attr_mask[i, self.settable_atom_attrs.index("charge")] = 0
            # If charge is not yet defined make sure there is room in the valence
            # Special case for N: adding charge to N increases its valence by 2, so we don't want to prevent that 
            # action, even if the max_valence is "full" (for v=3)
            if "charge" not in ad and explicit_valence[n] + 1 > (max_valence[n] + (2 if ad["v"] == "N" else 0)):
                s, e = self.atom_attr_logit_slice["charge"]
                set_node_attr_mask[i, s:e] = 0
            # idem for explicit hydrogens
            if "expl_H" not in ad and explicit_valence[n] + 1 > max_valence[n]:
                s, e = self.atom_attr_logit_slice["expl_H"]
                set_node_attr_mask[i, s:e] = 0

        remove_edge_mask = zeros((len(g.edges), 1))
        for i, e in enumerate(g.edges):
            if e not in bridges:
                remove_edge_mask[i] = 1
        edge_attr = zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = zeros((len(g.edges), self.num_edge_attr_logits))
        #TODO: go through all bonds, if connecting atoms are not editable, corresponding bond's mask is set to false
        remove_edge_attr_mask = zeros((len(g.edges), len(self.bond_attrs)))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for k, sl in zip(self.bond_attrs, self.bond_attr_slice):
                idx = self.bond_attr_values[k].index(ad[k]) if k in ad else 0
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
                if k in ad:  # If the attribute is already there, mask out logits
                    s, e = self.bond_attr_logit_slice[k]
                    set_edge_attr_mask[i, s:e] = 0
                    remove_edge_attr_mask[i, self.bond_attrs.index(k)] = 1
            # Check which bonds don't bust the valence of their atoms
            if "type" not in ad:  # Only if type isn't already set
                sl, _ = self.bond_attr_logit_slice["type"]
                for ti, bond_type in enumerate(self.bond_attr_values["type"][1:]):  # [1:] because 0th is default
                    # -1 because we'd be removing the single bond and replacing it with a double/triple/aromatic bond
                    is_ok = all([explicit_valence[n] + self._bond_valence[bond_type] - 1 <= max_valence[n] for n in e])
                    set_edge_attr_mask[i, sl + ti] = float(is_ok)
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=np.int64).reshape((-1, 2)).T

        if self.max_edges is not None and len(g.edges) >= self.max_edges:
            non_edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edges = set(g.edges)
            non_edge_index = np.array(
                [
                    (u, v)
                    for u in range(len(g))
                    for v in range(u + 1, len(g))
                    if (
                        (u, v) not in edges
                        and (v, u) not in edges
                        and explicit_valence[u] + 1 <= max_valence[u]
                        and explicit_valence[v] + 1 <= max_valence[v]
                    )
                ],
                dtype=np.int64,
            )
        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            non_edge_index=non_edge_index.astype(np.int64).reshape((-1, 2)).T,
            stop_mask=ones((1, 1)) * (len(g.nodes) > 0),  # Can only stop if there's at least a node
            add_node_mask=add_node_mask,
            set_node_attr_mask=set_node_attr_mask,
            add_edge_mask=ones((non_edge_index.shape[0], 1)),  # Already filtered by checking for valence
            set_edge_attr_mask=set_edge_attr_mask,
            remove_node_mask=remove_node_mask,
            remove_node_attr_mask=remove_node_attr_mask,
            remove_edge_mask=remove_edge_mask,
            remove_edge_attr_mask=remove_edge_attr_mask,
        )
        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()}, cond_info=cond_info, ft_cond_info = ft_cond_info)
        if 0:
            odata = C_data.as_torch()
            edge_reindexing = []
            for e in data.edge_index.T:
                for i, ep in enumerate(odata.edge_index.T):
                    if e[0] == ep[0] and e[1] == ep[1]:
                        edge_reindexing.append(i)
            edge_reindexing = torch.tensor(edge_reindexing).long().reshape(-1)
            non_edge_reindex = []
            for e in data.non_edge_index.T:
                for i, ep in enumerate(odata.non_edge_index.T):
                    if e[0] == ep[0] and e[1] == ep[1]:
                        non_edge_reindex.append(i)
            non_edge_reindex = torch.tensor(non_edge_reindex).long().reshape(-1)
            for k in odata.keys:
                fixed = getattr(odata, k)
                # if k == "non_edge_index":
                #     # let's just check they're all there
                #     assert set(map(tuple, fixed.T.tolist())) == set(map(tuple, data.non_edge_index.T.tolist()))
                #     continue
                if "edge" in k:
                    reindex = edge_reindexing
                    if k in ['add_edge_mask', 'non_edge_index']:
                        reindex = non_edge_reindex
                    if "index" in k:
                        fixed = fixed[:, reindex]
                    elif k == "edge_attr":
                        fixed = fixed[reindex]
                    else:
                        if fixed.shape[0] == 0:
                            assert fixed.shape == getattr(data, k).shape
                            continue
                        if k == 'add_edge_mask':
                            fixed = fixed[reindex]
                        else:
                            fixed = fixed[reindex[::2] // 2]
                assert (getattr(data, k) == fixed).all()

        if self.num_rw_feat > 0:
            data.x = torch.cat([data.x, random_walk_probs(data, self.num_rw_feat, skip_odd=True)], 1)
        #g._cached_Data = data
        return data

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        if self.graph_cls is not Graph:
            return Data_collate(graphs, ["edge_index", "non_edge_index"])
        if self.collate_split == 0:
            return gd.Batch.from_data_list(graphs, follow_batch=["edge_index", "non_edge_index"])
        else:
            idcs = np.linspace(0, len(graphs), self.collate_split + 1, dtype=np.int64)
            if not hasattr(self, "empty_graph"):
                self.empty_graph = copy.deepcopy(graphs[0])
                for k in self.empty_graph.keys:
                    if "edge_index" in k:
                        setattr(self.empty_graph, k, getattr(self.empty_graph, k)[:, :0])
                    else:
                        setattr(self.empty_graph, k, getattr(self.empty_graph, k)[:0])
            return BatchList(
                [
                    gd.Batch.from_data_list(
                        graphs[i:j] if j > i else [self.empty_graph], follow_batch=["edge_index", "non_edge_index"]
                    )
                    for i, j in zip(idcs, idcs[1:])
                ]
            )

    def mol_to_graph(self, mol: Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = self.graph_cls()
        mol = Mol(mol)  # Make a copy
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for a in mol.GetAtoms():
            attrs = {
                "chi": a.GetChiralTag(), #uncomment if using chirality
                "charge": a.GetFormalCharge(),
                "expl_H": a.GetNumExplicitHs(),
                # RDKit makes * atoms have no implicit Hs, but we don't want this to trickle down.
                "no_impl": a.GetNoImplicit() and a.GetSymbol() != "*",
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                **{attr: val for attr, val in attrs.items() if val != self.atom_attr_defaults[attr]},
                #**({"fill_wildcard": None} if a.GetSymbol() == "*" else {}),
            )
        for b in mol.GetBonds():
            attrs = {"type": b.GetBondType()}
            g.add_edge(
                b.GetBeginAtomIdx(),
                b.GetEndAtomIdx(),
                **{attr: val for attr, val in attrs.items() if val != self.bond_attr_defaults[attr]},
            )
        return g

    def graph_to_mol(self, g: Graph) -> Mol:
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            s = d.get("fill_wildcard", d["v"])
            a = Chem.Atom(s if s is not None else self.default_wildcard_replacement)
            if "chi" in d:
                a.SetChiralTag(d["chi"])
            if "charge" in d:
                a.SetFormalCharge(d["charge"])
            if "expl_H" in d:
                a.SetNumExplicitHs(d["expl_H"])
            if "no_impl" in d:
                a.SetNoImplicit(d["no_impl"])
            mp.AddAtom(a)
        for e in g.edges:
            d = g.edges[e]
            mp.AddBond(e[0], e[1], d.get("type", BondType.SINGLE))
        mp.CommitBatchEdit()
        Chem.SanitizeMol(mp)
        # Not sure why, but this seems to find errors that SanitizeMol doesn't, and it saves us trouble downstream,
        # since MolFromSmiles returns None if the SMILES encoding of the molecule is invalid, which seemingly happens
        # occasionally (although very rarely) even if RDKit is able to create a SMILES string from `mp`.
        return Chem.MolFromSmiles(Chem.MolToSmiles(mp))

    def is_sane(self, g: Graph) -> bool:
        try:
            mol = self.graph_to_mol(g)
        except Exception:
            return False
        if mol is None:
            return False
        return True