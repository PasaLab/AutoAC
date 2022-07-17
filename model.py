import copy
import numpy as np
import random
from collections import defaultdict
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from operations import *
# from torch.autograd import Variable
# from genotypes import PRIMITIVES
# from genotypes import Genotype

from utils.tools import *
from ops.operations import *
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AggrOp(nn.Module):
    def __init__(self, cluster_op_choice, valid_type, g, in_dim, out_dim, args):
        super(AggrOp, self).__init__()
        self.args = args
        self.g = g
        self._ops = nn.ModuleList()
        for op_name in cluster_op_choice:
            if op_name == 'one-hot':
                self._ops.append(None)
                continue
            op = OPS[op_name](valid_type, in_dim, out_dim, args)
            self._ops.append(op)
    
    def forward(self, mask_matrix, x, one_hot_h=None):
        # mask_matrix: no_grad; x: need_grad; weights: need_grad
        #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
        # print(f"weights shape: {weights.shape}")
        # print(f"mask_matrix shape: {mask_matrix.shape}")
        res = []
        for op in self._ops:
            if op is None:
                res.append(torch.spmm(mask_matrix, one_hot_h))
            else:
                res.append(torch.spmm(mask_matrix, op(self.g, x)))

        return sum(res)
        # return sum(torch.spmm(mask_matrix, op(self.g, x)) for op in self._ops)


class Network_discrete(nn.Module):
    def __init__(self, g, criterion, train_val_test, type_mask, dl, in_dims, num_classes, args, node_assign, alpha_params, e_feat):
        super(Network_discrete, self).__init__()
        # graph info
        self.g = g
        self._criterion = criterion
        self.dl = dl
        self.type_mask = type_mask
        self.e_feat = e_feat
        
        # train val test
        self.train_val_test = train_val_test
        self.train_idx, self.val_idx, self.test_idx = train_val_test[0], train_val_test[1], train_val_test[2]

        self.gnn_model_name = args.gnn_model
        self.in_dims = in_dims
        self.num_layers = args.num_layers
        self.num_classes = num_classes

        # GAT params
        self.heads = [args.num_heads] * args.num_layers + [1]
        self.dropout = args.dropout
        self.slope = args.slope

        self.cluster_num = args.cluster_num
        self.valid_attr_node_type = args.valid_attributed_type
        
        self.args = args

        # process discrete arch
        self.alpha_params = alpha_params
        self.node_assign = node_assign

        # record graph information
        self.all_nodes_num = dl.nodes['total']
        self.all_nodes_type_num = len(dl.nodes['count'])
        # print(f"node type num: {self.all_nodes_type_num}")

        self.node_type_split_list = [dl.nodes['count'][i] for i in range(len(dl.nodes['count']))]

        self.unAttributed_nodes_num = sum(1 for i in range(self.all_nodes_num) if not(dl.nodes['shift'][self.valid_attr_node_type] <= i <= dl.nodes['shift_end'][self.valid_attr_node_type]))
        # print(f"unAttributed nodes num: {self.unAttributed_nodes_num}")

        self.unAttributed_node_id_list = [i for i in range(self.all_nodes_num) if not(dl.nodes['shift'][self.valid_attr_node_type] <= i <= dl.nodes['shift_end'][self.valid_attr_node_type])]
        # print(f"self.unAttributed_node_id_list : {self.unAttributed_node_id_list}")
        # shuffle
        random.shuffle(self.unAttributed_node_id_list)

        self.clusternodeId2originId = {}
        self.originId2clusternodeId = {}
        for i, origin_id in enumerate(self.unAttributed_node_id_list):
            self.clusternodeId2originId[i] = origin_id
            self.originId2clusternodeId[origin_id] = i

        self.nodeid2type = {}
        for i in range(self.all_nodes_type_num):
            for j in range(dl.nodes['shift'][i], dl.nodes['shift_end'][i] + 1):
                self.nodeid2type[j] = i
        
        self._process_genotype()
        self._construct_mask_matrix()
        self._initialize_weights()
    
    def _process_genotype(self):
        # print(f"self.alpha_params:\n{self.alpha_params}")
        arch_weights = self.alpha_params
        arch_weights_softmax = softmax(self.alpha_params, axis=1)
        logger.info(f"arch_weights:\n{arch_weights}")
        logger.info(f"arch_weights_softmax:\n{arch_weights_softmax}")
        arch_indices = np.argmax(arch_weights, axis=1)
        self.cluster_op_choice = [PRIMITIVES[x] for x in arch_indices]
        logger.info(f"genotype choice:\n{self.cluster_op_choice}")

    def _construct_mask_matrix(self):
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            origin_id_arr = np.where(self.node_assign == i)[0]
            cur_cluster_node_id = [(x, x, 1) for x in origin_id_arr]
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))

    def _initialize_weights(self):
        initial_dim = self.in_dims[self.valid_attr_node_type]
        hidden_dim = self.args.hidden_dim
        self.preprocess = nn.Linear(initial_dim, hidden_dim, bias=True)
        # self.preprocess = nn.Linear(initial_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.preprocess.weight, gain=1.414)
        
        if 'one-hot' in PRIMITIVES:
            # construct one-hot embedding weight matrix
            self.one_hot_feature_list = []
            self.embedding_list = nn.ModuleList()
            for i in range(self.all_nodes_type_num):
                dim = self.node_type_split_list[i]
                if i == self.valid_attr_node_type:
                    self.one_hot_feature_list.append(None)
                    self.embedding_list.append(None)
                    continue
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                self.one_hot_feature_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
                self.embedding_list.append(nn.Linear(dim, hidden_dim, bias=True))
                nn.init.xavier_normal_(self.embedding_list[-1].weight, gain=1.414)

        if self.args.useTypeLinear:
            self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for i in range(self.all_nodes_type_num) if i != self.valid_attr_node_type])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        self._ops = nn.ModuleList()
        for k in range(self.cluster_num):
            op = AggrOp(self.cluster_op_choice, self.valid_attr_node_type, self.g, hidden_dim, hidden_dim, self.args)
            self._ops.append(op)
        
        self.gnn_model = self._get_gnn_model_func(self.gnn_model_name)
        # self.gnn_model = MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, hidden_dim, self.num_classes, self.num_layers, self.heads,
        #                         F.elu, self.dropout, self.dropout, self.slope, False)

    def _get_gnn_model_func(self, model_name):
        if model_name == 'gat':
            return MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, self.heads,
                                F.elu, self.dropout, self.dropout, self.slope, False, self.args.l2norm)
        elif model_name == 'gcn':
            return MODEL_NAME[self.gnn_model_name](self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, F.elu, self.args.dropout)
        elif model_name == 'simpleHGN':
            return MODEL_NAME[self.gnn_model_name](self.g, self.args.edge_feats, len(self.dl.links['count']) * 2 + 1, self.in_dims, self.args.hidden_dim, self.num_classes, self.num_layers, self.heads, F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.05)

    def _loss(self, x, y, is_valid=True):
        node_embedding, logits = self(x)
        if is_valid:
            input = logits[self.val_idx].cuda()
            target = y[self.val_idx].cuda()
        else:
            input = logits[self.train_idx].cuda()
            target = y[self.train_idx].cuda()         
        return self._criterion(input, target)

    def forward(self, features_list):
        # features attribute comletion learning
        h_raw_attributed_transform = self.preprocess(features_list[self.valid_attr_node_type])
        
        h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        raw_attributed_node_indices = np.where(self.type_mask == self.valid_attr_node_type)[0]
        h0[raw_attributed_node_indices] = h_raw_attributed_transform
        # h0 = torch.add(h0, h_raw_attributed_transform)

        # h = []
        # for feature in features_list:
        #     h.append(feature)
        # h = torch.cat(h, 0)
        # #TODO: zero vector meets problem? when back-propogation process
        # h0 = self.preprocess(h)
        # # h0 = F.elu(h0)
        
        one_hot_h = None
        if 'one-hot' in PRIMITIVES:
            # process one_hot_op
            one_hot_h = []
            for i in range(self.all_nodes_type_num):
                if i == self.valid_attr_node_type:
                    one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.hidden_dim)).to(device))
                    continue
                dense_h = self.embedding_list[i](self.one_hot_feature_list[i])
                one_hot_h.append(dense_h)
            one_hot_h = torch.cat(one_hot_h, 0)

        # h_attributed = None
        # h0 = F.dropout(h0, p=self.args.dropout)
        h_attributed = h0
        # h_attributed = F.elu(h0)
        for k in range(self.cluster_num):
            cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h0, one_hot_h)
            h_attributed = torch.add(h_attributed, cur_k_res)
            # if h_attributed is None:
            #     h_attributed = cur_k_res
            # else:
            #     h_attributed = torch.add(h_attributed, cur_k_res)
        
        if self.args.useTypeLinear:
            _h = h_attributed
            _h_list = torch.split(_h, self.node_type_split_list)

            h_transform = []
            fc_idx = 0
            for i in range(self.all_nodes_type_num):
                if i == self.valid_attr_node_type:
                    h_transform.append(_h_list[i])
                    continue
                h_transform.append(self.fc_list[fc_idx](_h_list[i]))
                fc_idx += 1
            h_transform = torch.cat(h_transform, 0)

            if self.args.usedropout:
                h_transform = F.dropout(h_transform, self.args.dropout)

            # gnn part
            node_embedding, logits = self.gnn_model(h_transform, self.e_feat)

        else:
            if self.args.usedropout:
                h_attributed = F.dropout(h_attributed, self.args.dropout)
            node_embedding, logits = self.gnn_model(h_attributed, self.e_feat)

        if self.args.dataset == 'IMDB':
            return node_embedding, logits, F.sigmoid(logits)
        else:
            return node_embedding, logits, logits