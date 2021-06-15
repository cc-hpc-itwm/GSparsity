#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07 #
#####################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from typing import Text
from torch.distributions.categorical import Categorical

from cell_operations import ResNetBasicblock, drop_path
from search_cells import NAS201SearchCell as SearchCell
from genotypes import Structure


class GenericNAS201Model(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(GenericNAS201Model, self).__init__()
        self._C = C
        self._layerN = N
        self._max_nodes = max_nodes
        self._stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self._cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self._cells.append(cell)
            C_prev = cell.out_dim
        self._op_names = deepcopy(search_space)
        self._Layer = len(self._cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(
                C_prev, affine=affine, track_running_stats=track_running_stats
            ),
            nn.ReLU(inplace=True),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._num_edge = num_edge

        self._mode = None
        self.dynamic_cell = None
        self._tau = None
        self._algo = None
        self._drop_path = None
        self.verbose = False

    
    def set_drop_path(self, progress, drop_path_rate):
        if drop_path_rate is None:
            self._drop_path = None
        elif progress is None:
            self._drop_path = drop_path_rate
        else:
            self._drop_path = progress * drop_path_rate

    @property
    def mode(self):
        return self._mode

    @property
    def drop_path(self):
        return self._drop_path
    """
    @property
    def weights(self):
        xlist = list(self._stem.parameters())
        xlist += list(self._cells.parameters())
        xlist += list(self.lastact.parameters())
        xlist += list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist
    """
    
    
    @property
    def message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self._cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self._cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={_max_nodes}, N={_layerN}, L={_Layer}, alg={_algo})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    @property
    def genotype(self):
        genotypes = []
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self._op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if use_random:
                    op_name = random.choice(self._op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self._op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def get_log_prob(self, arch):
        with torch.no_grad():
            logits = nn.functional.log_softmax(self.arch_parameters, dim=-1)
        select_logits = []
        for i, node_info in enumerate(arch.nodes):
            for op, xin in node_info:
                node_str = "{:}<-{:}".format(i + 1, xin)
                op_index = self._op_names.index(op)
                select_logits.append(logits[self.edge2index[node_str], op_index])
        return sum(select_logits).item()

    def return_topK(self, K, use_random=False):
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs):
            K = len(archs)
        if use_random:
            return random.sample(archs, K)
        else:
            sorted_pairs = sorted(pairs, key=lambda x: -x[0])
            return_pairs = [sorted_pairs[_][1] for _ in range(K)]
            return return_pairs

    def forward(self, inputs):
        feature = self._stem(inputs)
        for i, cell in enumerate(self._cells):
            feature = cell(feature)
            if self.drop_path is not None:
                feature = drop_path(feature, self.drop_path)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
