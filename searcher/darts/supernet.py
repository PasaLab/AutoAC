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

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

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
    def __init__(self):
        super(MixedOpShared, self).__init__()
        self._one_hot_idx = -1
        for i, primitive in enumerate(PRIMITIVES):
            if primitive == 'one-hot':
                self._one_hot_idx = i
                continue

    def forward(self, mask_matrix, h_op_list, weights=None):
        # mask_matrix: no_grad; x: need_grad; weights: need_grad
        #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
        res = []
        for w, op in zip(weights, h_op_list):
            if w.data > 0:
                res.append(w * torch.spmm(mask_matrix, op))
            else:
                res.append(w)
        return sum(res)


class Network_Darts(nn.Module):
    def __init__(self, data_info, idx_info, train_info, gnn_model_manager, args):
        super(Network_Darts, self).__init__()
        
        self.data_info = data_info
        self.idx_info = idx_info
        self.train_info = train_info
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        
        self.args = args
        self._logger = args.logger
        
        # self.gnn_model = ModelManager(data_info, idx_info, self.args)
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
        
        self._init_expectation_step()
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
        self.alphas = Variable(1e-3 * torch.randn(self.cluster_num, num_ops).cuda(), requires_grad=True)
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

        if self.args.useTypeLinear:
            feature_hidden_dim = self.args.hidden_dim
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
            for k in range(self.cluster_num):
                op = MixedOpShared()
                self._ops.append(op)
        else:
            self._ops = nn.ModuleList()
            for k in range(self.cluster_num):
                op = MixedOp(self.valid_attr_node_type, self.g, hidden_dim, hidden_dim, self.args)
                self._ops.append(op)

        if self.args.usebn:
            self.bn = nn.BatchNorm1d(self.args.hidden_dim)
        
        if self.args.use_dmon:
            self.transform_cluster_fc = nn.Linear(self.args.last_hidden_dim, self.cluster_num, bias=True)
            nn.init.xavier_normal_(self.transform_cluster_fc.weight, gain=1.414)
        
    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, x, y, minibatch_info=None, is_valid=True):
        if self.args.use_dmon:
            h_attribute, node_embedding, _, logits, dmon_loss, _ = self(x, minibatch_info)
        else:
            h_attribute, node_embedding, _, logits = self(x, minibatch_info)
        if is_valid:
            input = logits[self.val_idx].cuda()
            target = y[self.val_idx].cuda()
        else:
            input = logits[self.train_idx].cuda()
            target = y[self.train_idx].cuda()
        
        if self.args.use_dmon:
            return self._criterion(input, target) + dmon_loss * self.args.dmon_loss_alpha
        else:
            return self._criterion(input, target)
    
    def _loss_minibatch(self, x, y, minibatch_info=None, _node_embedding=None, is_valid=True):
        if self.args.use_dmon:
            h_attribute, node_embedding, _, logits, dmon_loss, _ = self(x, minibatch_info)
        else:
            h_attribute, node_embedding, _, logits = self(x, minibatch_info)
        _, _, _, idx_batch = minibatch_info
        _t = time.time()
        _node_embedding = scatter_embbeding(_node_embedding, h_attribute, node_embedding, idx_batch)
        # logger.info(f"val scatter_embbeding time: {time.time() - _t} ")
        input = logits.cuda()
        # input = logits[idx_batch].cuda()
        target = y[idx_batch].cuda()  
        if self.args.use_dmon:
            return self._criterion(input, target) + dmon_loss * self.args.dmon_loss_alpha, _node_embedding
        else:
            return self._criterion(input, target), _node_embedding
    
    def execute_maximum_step(self, node_embedding):
        # node_emb = node_embedding.item()
        node_emb = node_embedding.detach().cpu().numpy()

        assert node_emb.shape[0] == self.all_nodes_num

        unAttributed_node_emb = []
        for i in range(self.unAttributed_nodes_num):
            # print(i)
            origin_idx = self.clusternodeId2originId[i]
            unAttributed_node_emb.append(node_emb[origin_idx].tolist())
        unAttributed_node_emb = np.array(unAttributed_node_emb)

        if self.args.cluster_norm:
            # scale to (0, 1)
            unAttributed_node_emb = preprocessing.scale(unAttributed_node_emb)

        new_centers = np.array([unAttributed_node_emb[self.node_cluster_class == j, :].mean(axis=0) for j in range(self.cluster_num)])

        return unAttributed_node_emb, new_centers

    def execute_expectation_step(self, unAttributed_node_emb, new_centers):
        new_assign = np.argmin(((unAttributed_node_emb[:, :, None] - new_centers.T[None, :, :]) ** 2).sum(axis=1), axis=1)
        self.node_cluster_class = copy.deepcopy(new_assign)
        self._update_cluster_info()
        return self._gen_cluster_info()
    
    def _gen_cluster_info(self):
        node_cluster_class = self.node_cluster_class
        info_str = ""
        origin_id_cluster_dict = [-1] * self.all_nodes_num
        for i in range(len(node_cluster_class)):
            original_id = self.clusternodeId2originId[i]
            origin_id_cluster_dict[original_id] = node_cluster_class[i]

        for i in range(self.all_nodes_num):
            info_str += str(i) + ': ' + str(origin_id_cluster_dict[i]) + ';\t'

        return info_str, origin_id_cluster_dict

    def _update_cluster_info(self):
        # empty and update
        for k in range(self.cluster_num):
            self.clusters[k]['node_id'] = []
        for i in range(self.unAttributed_nodes_num):
            self.clusters[self.node_cluster_class[i]]['node_id'].append(i)
        
        # mask matrix for each cluster
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            cur_cluster_node_id = [(self.clusternodeId2originId[x], self.clusternodeId2originId[x], 1) for x in self.clusters[i]['node_id']]
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))
        return self.clusters

    def create_new_assignment(self, assignments):
        new_assign = np.argmax(assignments, axis=1)
        self.node_cluster_class = copy.deepcopy(new_assign)
        self._update_cluster_info_dmon()
        return self._gen_cluster_info_dmon()
    
    def _update_cluster_info_dmon(self):
        # empty and update
        for k in range(self.cluster_num):
            self.clusters[k]['node_id'] = []
        for i in range(self.all_nodes_num):
            if i in self.unAttributed_node_id_list_copy:
                self.clusters[self.node_cluster_class[i]]['node_id'].append(i)
        
        # mask matrix for each cluster
        self.cluster_mask_matrix = []
        for i in range(self.cluster_num):
            # cur_cluster_node_id = [(self.unAttributedID2originID[x], self.unAttributedID2originID[x], 1) for x in self.clusters[i]['node_id']]
            cur_cluster_node_id = [(x, x, 1) for x in self.clusters[i]['node_id']]
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))
        return self.clusters
    
    def _gen_cluster_info_dmon(self):
        node_cluster_class = self.node_cluster_class
        info_str = ""
        origin_id_cluster_dict = [-1] * self.all_nodes_num
        for i in range(len(node_cluster_class)):
            # original_id = self.unAttributedID2originID[i]
            # origin_id_cluster_dict[original_id] = node_cluster_class[i]
            if i in self.unAttributed_node_id_list_copy:
                origin_id_cluster_dict[i] = node_cluster_class[i]
        
        for i in range(self.all_nodes_num):
            info_str += str(i) + ': ' + str(origin_id_cluster_dict[i]) + ';\t'

        return info_str, origin_id_cluster_dict

    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
            self.saved_params[index].copy_(value.data)

    def step(self, X, y, minibatch_info_train=None, minibatch_info_val=None, eta=None, network_optimizer=None, arch_optimizer=None, unrolled=False, _node_embedding=None):
        arch_optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(X, y, minibatch_info_train, minibatch_info_val, _node_embedding, eta, network_optimizer)
        else:
            _node_embedding = self._backward_step(X, y, minibatch_info_val, _node_embedding, is_valid=True)
            
        arch_optimizer.step()
        
        if minibatch_info_train is not None:
            return _node_embedding

    def _backward_step(self, X, y, minibatch_info, _node_embedding=None, is_valid=True):
        if minibatch_info is None:
            loss = self._loss(X, y, minibatch_info, is_valid)
        else:
            loss, _node_embedding = self._loss_minibatch(X, y, minibatch_info, _node_embedding, is_valid)
        loss.backward()

        if minibatch_info is not None:
            return _node_embedding

    def _backward_step_unrolled(self, X, y, minibatch_info_train, minibatch_info_val, _node_embedding, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(X, y, minibatch_info_train, eta, network_optimizer)
        if minibatch_info_val is None:
            unrolled_loss = unrolled_model._loss(X, y, minibatch_info_val, is_valid=True)
        else:
            unrolled_loss, _node_embedding = unrolled_model._loss_minibatch(X, y, minibatch_info_val)
        # unrolled_loss = unrolled_model._loss(X, y, is_valid=True) # validation loss
        unrolled_loss.backward() # one-step update for w?
        dalpha = [v.grad for v in unrolled_model.arch_parameters()] #L_vali w.r.t alpha
        # for name, paramer in unrolled_model.named_parameters():
        #     self._logger.info(f"{name}, {paramer.grad}")
        vector = [v.grad.data for v in unrolled_model.parameters()] # gradient, L_train w.r.t w, double check the model construction
        implicit_grads = self._hessian_vector_product(vector, X, y, minibatch_info_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        #update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        return _node_embedding
    
    def _compute_unrolled_model(self, X, y, minibatch_info, eta, network_optimizer):
        if minibatch_info is None:
            loss = self._loss(X, y, minibatch_info, is_valid=False)
        else:
            loss, _node_embedding = self._loss_minibatch(X, y, minibatch_info)
        theta = _concat(self.parameters()).data# w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.parameters()).mul_(self.args.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # for name, paramer in self.named_parameters():
        #     self._logger.info(f"{name}, {paramer.grad}")
        # print(f"self.model.parameters: {self.model.parameters()}")
        dtheta = _concat(torch.autograd.grad(loss, self.parameters())).data + self.args.weight_decay * theta#gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta)) # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()

        params, offset = {}, 0
        for k, v in self.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()
    
    def new(self):
        model_copy = Network_Darts(self.data_info, self.idx_info, self.train_info, self.gnn_model_manager, self.args).cuda()
        for x, y in zip(model_copy.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_copy
    
    def _hessian_vector_product(self, vector, X, y, minibatch_info, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v) # R * d(L_val/w', i.e., get w^+
        if minibatch_info is None:
            loss = self._loss(X, y, minibatch_info, is_valid=False)
        else:
            loss, _node_embedding = self._loss_minibatch(X, y, minibatch_info)
        # loss = self.model._loss(X, y, is_valid=False) # train loss
        grads_p = torch.autograd.grad(loss, self.arch_parameters()) # d(L_train)/d_alpha, w^+

        for p, v in zip(self.parameters(), vector):
            p.data.sub_(2 * R, v) # get w^-, need to subtract 2 * R since it has add R
        if minibatch_info is None:
            loss = self._loss(X, y, minibatch_info, is_valid=False)
        else:
            loss, _node_embedding = self._loss_minibatch(X, y, minibatch_info)       
        # loss = self.model._loss(X, y, is_valid=False)# train loss
        grads_n = torch.autograd.grad(loss, self.arch_parameters())# d(L_train)/d_alpha, w^-

        #reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    def process_cluster(self, h):
        # self._logger.info(f"h shape: {h.shape}")
        # assignments = F.softmax(self.transform_cluster_fc(h_unattribute), dim=1)
        assignments = F.softmax(self.transform_cluster_fc(h), dim=1)
        cluster_sizes = torch.sum(assignments, dim=0)  # Size [k].
        assignments_pooling = assignments / cluster_sizes  # Size [n, k].
        
        adjacency = self.adjM
        # adjacency = self.g.adjacency_matrix()
        
        degrees = torch.sparse.sum(adjacency, dim=0)
        degrees = degrees.to_dense()
        degrees = torch.reshape(degrees, (-1, 1))
        
        number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(degrees)

        # self._logger.info(f"adjacency shape: {adjacency.shape}")
        # self._logger.info(f"assignments shape: {assignments.shape}")
        
        # Computes the size [k, k] pooled graph as S^T*A*S in two multiplications.
        graph_pooled = torch.transpose(
            torch.spmm(adjacency, assignments), 0, 1)
        graph_pooled = torch.matmul(graph_pooled, assignments)

        # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
        # in three matrix multiplications by first processing the left part S^T*d
        # and then multyplying it by the right part d^T*S.
        # Left part is [k, 1] tensor.
        normalizer_left = torch.matmul(assignments.T, degrees)
        # Right part is [1, k] tensor.
        normalizer_right = torch.matmul(degrees.T, assignments)

        # Normalizer is rank-1 correction for degree distribution for degrees of the
        # nodes in the original graph, casted to the pooled graph.
        normalizer = torch.matmul(normalizer_left,
                            normalizer_right) / 2 / number_of_edges
        spectral_loss = -torch.trace(graph_pooled - normalizer) / 2 / number_of_edges
        # self.add_loss(spectral_loss)

        collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * math.sqrt(
            float(self.cluster_num)) - 1
        
        dmon_loss = spectral_loss + self.args.collapse_regularization * collapse_loss
        
        return dmon_loss, assignments
    
    def forward(self, features_list, mini_batch_input=None, use_dmon=False):
        # features attribute comletion learning
        h_raw_attributed_transform = self.preprocess(features_list[self.valid_attr_node_type])
        # h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        h0 = torch.zeros(self.all_nodes_num, self.args.att_comp_dim, device=device)
        raw_attributed_node_indices = np.where(self.type_mask == self.valid_attr_node_type)[0]
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

        self.alphas_weight = F.softmax(self.alphas, dim=-1)
        
        if self.args.shared_ops:
            h_op_list = []
            for op in self._shared_op:
                if op is None:
                    h_op_list.append(one_hot_h)
                else:
                    h_op = op(self.g, h0)
                    h_op_list.append(h_op)
            h_attributed = None
            for k in range(self.cluster_num):
                cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h_op_list, self.alphas_weight[k])
                if h_attributed is None:
                    h_attributed = cur_k_res
                else:
                    h_attributed = torch.add(h_attributed, cur_k_res)
            h_attributed = torch.add(h_attributed, h0)
        else:
            h_attributed = None
            for k in range(self.cluster_num):
                cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h0, one_hot_h, self.alphas_weight[k])
                if h_attributed is None:
                    h_attributed = cur_k_res
                else:
                    h_attributed = torch.add(h_attributed, cur_k_res)
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
            
            # self._logger.info(f"h_transform shape: {h_transform.shape}")
            # gnn part
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_transform, mini_batch_input)

        else:
        
            if self.args.usebn:
                h_attributed = self.bn(h_attributed)
                
            if self.args.usedropout:
                h_attributed = F.dropout(h_attributed, self.args.dropout)
            
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_attributed, mini_batch_input)
        
        if self.args.use_dmon:
            if self.args.use_minibatch:
                _node_embedding = None
                _, _, _, idx_batch = mini_batch_input
                _node_embedding = scatter_add(_node_embedding, h_attributed, node_embedding, idx_batch)
            else:
                _node_embedding = node_embedding
            
            dmon_loss, assignments = self.process_cluster(_node_embedding)
            if self.args.dataset == 'IMDB':
                return h_attributed, node_embedding, logits, F.sigmoid(logits), dmon_loss, assignments
            else:
                return h_attributed, node_embedding, logits, logits, dmon_loss, assignments
        
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
        return F.softmax(self.arch_parameters()[0], dim=-1)
    