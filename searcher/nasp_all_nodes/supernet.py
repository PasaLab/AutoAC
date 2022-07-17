import copy
import numpy as np
import random
from collections import defaultdict
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import math

from utils.tools import *
from ops.operations import *
from models import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MixedOp(nn.Module):
    def __init__(self, valid_type, g, in_dim, out_dim, args):
        super(MixedOp, self).__init__()
        self.g = g
        self._ops = nn.ModuleList()
        self._one_hot_idx = -1
        for i, primitive in enumerate(PRIMITIVES):
            if primitive == 'one-hot':
                self._one_hot_idx = i
                self._ops.append(None)
                continue

            op = OPS[primitive](valid_type, in_dim, out_dim, args)
            self._ops.append(op)

    def forward(self, mask_matrix, x, one_hot_h=None, weights=None):
        # mask_matrix: no_grad; x: need_grad; weights: need_grad
        #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
        # print(f"weights shape: {weights.shape}")
        # print(f"mask_matrix shape: {mask_matrix.shape}")
        res = []
        idx = 0
        for w, op in zip(weights, self._ops):
            if w.data > 0:
                if idx == self._one_hot_idx:
                    res.append(w * torch.spmm(mask_matrix, one_hot_h))
                else:
                    res.append(w * torch.spmm(mask_matrix, op(self.g, x)))
            else:
                res.append(w)
            idx += 1
        return sum(res)


class MixedOpShared(nn.Module):
    def __init__(self, original_id):
        super(MixedOpShared, self).__init__()
        self._one_hot_idx = -1
        self.original_id = original_id
        for i, primitive in enumerate(PRIMITIVES):
            if primitive == 'one-hot':
                self._one_hot_idx = i
                continue

    def forward(self, args, h_op_list, weights=None):
        # mask_matrix: no_grad; x: need_grad; weights: need_grad
        #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
        res = []
        for w, op in zip(weights, h_op_list):
            # args.logger.info(f"op size: {op.size()}")
            if w.data > 0:
                res.append(w * op[self.original_id])
            else:
                res.append(w)
        
        return sum(res)


class Network_Nasp_Allnodes(nn.Module):
    def __init__(self, data_info, idx_info, train_info, gnn_model_manager, args):
        super(Network_Nasp_Allnodes, self).__init__()
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        
        self.args = args
        self._logger = args.logger
        
        self.gnn_model_manager = gnn_model_manager
        self.gnn_model = self.gnn_model_manager.create_model_class()
        
        # data_info, idx_info, train_info = self.gnn_model.get_graph_info()
        
        self.gnn_model_name = args.gnn_model
        self.num_layers = args.num_layers

        # GAT params
        self.heads = [args.num_heads] * args.num_layers + [1]
        self.dropout = args.dropout
        self.slope = args.slope

        self.cluster_num = args.cluster_num
        self.valid_attr_node_type = args.valid_attributed_type

        # record graph information
        self.all_nodes_num = self.dl.nodes['total']
        self.all_nodes_type_num = len(self.dl.nodes['count'])
        self.node_type_split_list = [self.dl.nodes['count'][i] for i in range(len(self.dl.nodes['count']))]

        self.unAttributed_nodes_num = sum(1 for i in range(self.all_nodes_num) if not(self.dl.nodes['shift'][self.valid_attr_node_type] <= i <= self.dl.nodes['shift_end'][self.valid_attr_node_type]))

        self.unAttributed_node_id_list = [i for i in range(self.all_nodes_num) if not(self.dl.nodes['shift'][self.valid_attr_node_type] <= i <= self.dl.nodes['shift_end'][self.valid_attr_node_type])]
        self.unAttributed_node_id_list_copy = copy.deepcopy(self.unAttributed_node_id_list)
        
        self.unAttributedID2nodeID = {}
        for i, origin_id in enumerate(self.unAttributed_node_id_list):
            self.unAttributedID2nodeID[i] = origin_id
            
        # shuffle
        random.shuffle(self.unAttributed_node_id_list)

        self.clusternodeId2originId = {}
        self.originId2clusternodeId = {}
        for i, origin_id in enumerate(self.unAttributed_node_id_list):
            self.clusternodeId2originId[i] = origin_id
            self.originId2clusternodeId[origin_id] = i

        self.nodeid2type = {}
        for i in range(self.all_nodes_type_num):
            for j in range(self.dl.nodes['shift'][i], self.dl.nodes['shift_end'][i] + 1):
                self.nodeid2type[j] = i
        
        adjM = self.g.adjacency_matrix()
        self.adjM = adjM.to(device=device)
        
        # self._init_expectation_step()
        self._initialize_alphas()
        self._initialize_weights()

        self.saved_params = []
        for w in self._arch_parameters:
            temp = w.data.clone()
            self.saved_params.append(temp)
    
    def _init_expectation_step(self):
        # node id assign to cluster
        avg_node_num = int(self.unAttributed_nodes_num // self.cluster_num)
        remain_node_num = self.unAttributed_nodes_num % self.cluster_num
        self.init_cluster_params = {
            'each_cluster_node_num': avg_node_num,
            'last_cluster_node_num': avg_node_num + remain_node_num
        }

        temp_unAttributed_node_id_list = copy.deepcopy(self.unAttributed_node_id_list)

        self.clusters = []
        self.node_cluster_class = [0] * self.unAttributed_nodes_num
        
        shift = 0
        for i in range(self.cluster_num):
            if i < self.cluster_num - 1:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, shift + avg_node_num))
            else:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, self.unAttributed_nodes_num))
            
            # assign the node id to its cluster-class
            for idx in self.clusters[i]['node_id']:
                # print(idx)
                self.node_cluster_class[idx] = i

            shift += avg_node_num
        
        self.node_cluster_class = np.array(self.node_cluster_class)

        # mask matrix for each cluster
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            cur_cluster_node_id = [(self.clusternodeId2originId[x], self.clusternodeId2originId[x], 1) for x in self.clusters[i]['node_id']]
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        # self.alphas = Variable(1e-3 * torch.randn(self.cluster_num, num_ops).cuda(), requires_grad=True)
        # self.alphas = Variable(torch.ones(self.cluster_num, num_ops).cuda() / 2, requires_grad=True)
        self.alphas = Variable(torch.ones(self.unAttributed_nodes_num, num_ops).cuda() / 2, requires_grad=True)
        self._arch_parameters = [self.alphas]
        
    def _initialize_weights(self):
        initial_dim = self.in_dims[self.valid_attr_node_type]
        # hidden_dim = self.args.hidden_dim
        hidden_dim = self.args.att_comp_dim
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
                # self.embedding_list.append(nn.Linear(dim, hidden_dim, bias=False))
                nn.init.xavier_normal_(self.embedding_list[-1].weight, gain=1.414)

        feature_hidden_dim = self.args.hidden_dim
        
        if self.args.useTypeLinear:
            # self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, feature_hidden_dim, bias=True) for i in range(self.all_nodes_type_num) if i != self.valid_attr_node_type])
            self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, feature_hidden_dim, bias=True) for i in range(self.all_nodes_type_num)])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        if self.args.shared_ops:
            self._shared_op = nn.ModuleList()
            for primitive in PRIMITIVES:
                if primitive == 'one-hot':
                    cur_op_matrix = None
                else:
                    cur_op_matrix = OPS[primitive](self.valid_attr_node_type, hidden_dim, hidden_dim, self.args)
                self._shared_op.append(cur_op_matrix)
            self._ops = nn.ModuleList()
            # for k in range(self.cluster_num):
            # for k in range(len(PRIMITIVES)):
            for k in range(self.unAttributed_nodes_num):
                original_id = self.unAttributedID2nodeID[k]
                op = MixedOpShared(original_id)
                self._ops.append(op)
        
        if self.args.usebn:
            self.bn = nn.BatchNorm1d(self.args.hidden_dim)

        if self.args.use_skip:
            self.res_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2, bias=True),
                # nn.ReLU(),
                nn.ELU(),
                nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
            )
            for w in self.res_fc:
                if isinstance(w, nn.Linear):
                    nn.init.xavier_normal_(w.weight, gain=1.414)
            # self.res_fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
            # nn.init.xavier_normal_(self.res_fc.weight, gain=1.414)
            
    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, x, y, minibatch_info=None, is_valid=True):
        h_attribute, node_embedding, _, logits = self(x, minibatch_info)
        if is_valid:
            input = logits[self.val_idx].cuda()
            target = y[self.val_idx].cuda()
        else:
            input = logits[self.train_idx].cuda()
            target = y[self.train_idx].cuda()
        
        return self._criterion(input, target)
    
    def _loss_minibatch(self, x, y, minibatch_info=None, _node_embedding=None):
        h_attribute, node_embedding, _, logits = self(x, minibatch_info)
        _, _, _, idx_batch = minibatch_info
        _t = time.time()
        # _node_embedding = scatter_embbeding(_node_embedding, h_attribute, node_embedding, idx_batch)
        # logger.info(f"val scatter_embbeding time: {time.time() - _t} ")
        input = logits.cuda()
        # input = logits[idx_batch].cuda()
        target = y[idx_batch].cuda()  
        return self._criterion(input, target), _node_embedding
    

    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
            self.saved_params[index].copy_(value.data)

    def clip(self):
        clip_scale = []
        # 大于1和小于0的都置为1和0，中间的不动
        m = nn.Hardtanh(0, 1)
        for index in range(len(self._arch_parameters)):
            clip_scale.append(m(Variable(self._arch_parameters[index].data)))
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = clip_scale[index].data

    def proximal_step(self, var, maxIndexs=None):
        values = var.data.cpu().numpy()
        m, n = values.shape
        alphas = []
        # 对\alpha二值化
        for i in range(m):
            for j in range(n):
                if j == maxIndexs[i]:
                    # 提前保存一下\arch_parameters里每个layer最大\alpha值，然后把这个values都做二值化
                    alphas.append(values[i][j].copy())
                    values[i][j] = 1
                else:
                    values[i][j] = 0
        
        return torch.Tensor(values).cuda()

    def binarization(self, e_greedy=0):
        self.save_params()
        for index in range(len(self._arch_parameters)):
            m, n = self._arch_parameters[index].size()
            # 随机为每个layer选一个op
            if np.random.rand() <= e_greedy:
                maxIndexs = np.random.choice(range(n), m)
            else:
                maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)
            self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index], maxIndexs)

    def restore(self):
        # 更新\alpha参数前做二值化，梯度下降完了之后再还原成二值化之前的
        for index in range(len(self._arch_parameters)):
            self._arch_parameters[index].data = self.saved_params[index]

    def step(self, X, y, minibatch_info=None, eta=None, arch_optimizer=None, _node_embedding=None):
        arch_optimizer.zero_grad()
        self.binarization(self.args.e_greedy)
        
        # self._logger.info("=========== before step =============")
        # for name,parameters in self.named_parameters():
        #     self._logger.info(f"{name}':' {parameters}")
            
        if minibatch_info is None:
            loss = self._loss(X, y, minibatch_info)
        else:
            loss, _node_embedding = self._loss_minibatch(X, y, minibatch_info, _node_embedding)

        loss.backward()
        self.restore()
        arch_optimizer.step()
        
        # self._logger.info("=========== after step =============")
        # for name,parameters in self.named_parameters():
        #     self._logger.info(f"{name}':' {parameters}")
            
        if minibatch_info is not None:
            return _node_embedding

    def forward(self, features_list, mini_batch_input=None, use_dmon=False):
        # features attribute comletion learning
        h_raw_attributed_transform = self.preprocess(features_list[self.valid_attr_node_type])
        # h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        h0 = torch.zeros(self.all_nodes_num, self.args.att_comp_dim, device=device)
        raw_attributed_node_indices = np.where(self.type_mask == self.valid_attr_node_type)[0]
        # self._logger.info(f"h0 size: {h0.size()} h_raw_attributed_transform size: {h_raw_attributed_transform.size()}")
        # self._logger.info(f"raw_attributed_node_indices: {raw_attributed_node_indices}")
        
        h0[raw_attributed_node_indices] = h_raw_attributed_transform

        one_hot_h = None
        if 'one-hot' in PRIMITIVES:
            # process one_hot_op
            one_hot_h = []
            for i in range(self.all_nodes_type_num):
                if i == self.valid_attr_node_type:
                    # one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.hidden_dim)).to(device))
                    one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.att_comp_dim)).to(device))
                    continue
                dense_h = self.embedding_list[i](self.one_hot_feature_list[i])
                one_hot_h.append(dense_h)
            one_hot_h = torch.cat(one_hot_h, 0)

        if self.args.shared_ops:
            h_op_list = []
            for op in self._shared_op:
                if op is None:
                    h_op_list.append(one_hot_h)
                else:
                    h_op = op(self.g, h0)
                    h_op_list.append(h_op)
            h_attributed = torch.zeros(self.all_nodes_num, self.args.att_comp_dim, device=device)
            # for k in range(self.cluster_num):
            # for k in range(len(PRIMITIVES)):
            sset = {}
            for k in range(self.unAttributed_nodes_num):
                origin_nodeid = self.unAttributedID2nodeID[k]
                cur_k_res = self._ops[k](self.args, h_op_list, self._arch_parameters[0][k])
                # if origin_nodeid in sset:
                #     self._logger.info(f"nonono")
                # else:
                #     sset[origin_nodeid] = 1
                # self._logger.info(f"cur_k_res size: {cur_k_res.size()}")
                h_attributed[origin_nodeid] = cur_k_res

            h_attributed = torch.add(h_attributed, h0)
            
        if self.args.useTypeLinear:
            _h = h_attributed
            _h_list = torch.split(_h, self.node_type_split_list)

            h_transform = []
            fc_idx = 0
            for i in range(self.all_nodes_type_num):
                # if i == self.valid_attr_node_type:
                #     h_transform.append(_h_list[i])
                #     continue
                h_transform.append(self.fc_list[fc_idx](_h_list[i]))
                fc_idx += 1
            h_transform = torch.cat(h_transform, 0)

            if self.args.usedropout:
                h_transform = F.dropout(h_transform, self.args.dropout)

            # if self.args.use_skip:
            #     # h_attributed = h_attributed + self.res_fc(h_attributed)
            #     h_attributed = F.elu(h_attributed + self.res_fc(h_attributed))
            
            # self._logger.info(f"h_transform shape: {h_transform.shape}")
            # gnn part
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_transform, mini_batch_input)
        else:
            if self.args.usebn:
                h_attributed = self.bn(h_attributed)
            
            if self.args.usedropout:
                h_attributed = F.dropout(h_attributed, self.args.dropout)
            
            # if self.args.use_skip:
            #     # h_attributed = h_attributed + self.res_fc(h_attributed)
            #     h_attributed = F.elu(h_attributed + self.res_fc(h_attributed))
            
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_attributed, mini_batch_input)

        if self.args.dataset == 'IMDB':
            return h_attributed, node_embedding, logits, F.sigmoid(logits)
        else:
            return h_attributed, node_embedding, logits, logits

    def genotype(self):
        def _parse(arch_weights):
            gene = []
            arch_indices = torch.argmax(arch_weights, dim=-1)
            for k in arch_indices:
                gene.append(PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu())
        
    def print_alpha_params(self):
        return self.arch_parameters()[0]
    