import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
import random

from utils.tools import *
from ops.operations import *
from models import *

# class AggrOp(nn.Module):
#     def __init__(self, cluster_op_choice, valid_type, g, in_dim, out_dim, args):
#         super(AggrOp, self).__init__()
#         self.args = args
#         self.g = g
#         self._ops = nn.ModuleList()
#         for op_name in cluster_op_choice:
#             if op_name == 'one-hot':
#                 self._ops.append(None)
#                 continue
#             op = OPS[op_name](valid_type, in_dim, out_dim, args)
#             self._ops.append(op)
    
#     def forward(self, mask_matrix, x, one_hot_h=None):
#         # mask_matrix: no_grad; x: need_grad; weights: need_grad
#         #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
#         res = []
#         for op in self._ops:
#             if op is None:
#                 res.append(torch.spmm(mask_matrix, one_hot_h))
#             else:
#                 res.append(torch.spmm(mask_matrix, op(self.g, x)))

#         return sum(res)
#         # return sum(torch.spmm(mask_matrix, op(self.g, x)) for op in self._ops)

# class AggrOpShared(nn.Module):
#     def __init__(self, cluster_op_choice, valid_type, g, in_dim, out_dim, args):
#         super(AggrOp, self).__init__()
#         self.args = args
#         self.g = g
#         self._ops = nn.ModuleList()
#         for op_name in cluster_op_choice:
#             if op_name == 'one-hot':
#                 self._ops.append(None)
#                 continue
#             op = OPS[op_name](valid_type, in_dim, out_dim, args)
#             self._ops.append(op)
    
#     def forward(self, mask_matrix, x, one_hot_h=None):
#         # mask_matrix: no_grad; x: need_grad; weights: need_grad
#         #TODO: this place will need use torch.select to select the correspoding indx if this one cost much
#         res = []
#         for op in self._ops:
#             if op is None:
#                 res.append(torch.spmm(mask_matrix, one_hot_h))
#             else:
#                 res.append(torch.spmm(mask_matrix, op(self.g, x)))

#         return sum(res)
#         # return sum(torch.spmm(mask_matrix, op(self.g, x)) for op in self._ops)


class FixedNet(nn.Module):
    def __init__(self, data_info, idx_info, train_info, inner_data_info, gnn_model_manager, alpha, node_assign, args):
        super(FixedNet, self).__init__()
        
        self.args = args
        self._logger = args.logger
        
        self.data_info = data_info
        self.idx_info = idx_info
        self.train_info = train_info
        self.inner_data_info =  inner_data_info
        self.gnn_model_manager = gnn_model_manager
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        
        self.alpha = alpha
        self.node_assign = node_assign
        
        # record graph information
        self.all_nodes_num = self.dl.nodes['total']
        self.all_nodes_type_num = len(self.dl.nodes['count'])
        # print(f"node type num: {self.all_nodes_type_num}")

        self.node_type_split_list = [self.dl.nodes['count'][i] for i in range(len(self.dl.nodes['count']))]

        self.unAttributed_nodes_num = sum(1 for i in range(self.all_nodes_num) if not(self.dl.nodes['shift'][self.args.valid_attributed_type] <= i <= self.dl.nodes['shift_end'][self.args.valid_attributed_type]))
        # print(f"unAttributed nodes num: {self.unAttributed_nodes_num}")

        self.unAttributed_node_id_list = [i for i in range(self.all_nodes_num) if not(self.dl.nodes['shift'][self.args.valid_attributed_type] <= i <= self.dl.nodes['shift_end'][self.args.valid_attributed_type])]
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
            for j in range(self.dl.nodes['shift'][i], self.dl.nodes['shift_end'][i] + 1):
                self.nodeid2type[j] = i
        
        self._process_genotype()
        self._construct_mask_matrix()
        self._initialize_weights()
        
    def _process_genotype(self):
        arch_weights = self.alpha
        arch_weights_softmax = softmax(self.alpha, axis=1)
        self._logger.info(f"arch_weights:\n{arch_weights}")
        self._logger.info(f"arch_weights_softmax:\n{arch_weights_softmax}")
        arch_indices = np.argmax(arch_weights, axis=1)
        self.cluster_op_choice = [PRIMITIVES[x] for x in arch_indices]
        self._logger.info(f"genotype choice:\n{self.cluster_op_choice}")
        
    def _construct_mask_matrix(self):
        self.cluster_mask_matrix = []
        for i in range(self.args.cluster_num):
            origin_id_arr = np.where(self.node_assign == i)[0]
            cur_cluster_node_id = [(x, x, 1) for x in origin_id_arr]
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))

    def _initialize_weights(self):
        initial_dim = self.in_dims[self.args.valid_attributed_type]
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
                if i == self.args.valid_attributed_type:
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
            feature_hidden_dim = self.args.hidden_dim
            # self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for i in range(self.all_nodes_type_num) if i != self.args.valid_attributed_type])
            self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, feature_hidden_dim, bias=True) for i in range(self.all_nodes_type_num)])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        if self.args.shared_ops:
            self._shared_ops = nn.ModuleList()
            self._op_name_list = list(set(self.cluster_op_choice))
            for op_name in self._op_name_list:
                if op_name == 'one-hot':
                    op = None
                else:
                    op = OPS[op_name](self.args.valid_attributed_type, hidden_dim, hidden_dim, self.args)
                self._shared_ops.append(op)
        else:
            self._ops = nn.ModuleList()
            for k in range(self.args.cluster_num):
                cur_optim_name = self.cluster_op_choice[k]
                if cur_optim_name == 'one-hot':
                    op = None
                else:
                    op = OPS[cur_optim_name](self.args.valid_attributed_type, hidden_dim, hidden_dim, self.args)
                self._ops.append(op)
                
            # for k in range(self.args.cluster_num):
            #     op = AggrOp(self.cluster_op_choice, self.args.valid_attributed_type, self.g, hidden_dim, hidden_dim, self.args)
            #     self._ops.append(op)

        self.gnn_model = self.gnn_model_manager.create_model_class()
        
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
        
    def forward(self, features_list, mini_batch_input=None):
            # features attribute comletion learning
        h_raw_attributed_transform = self.preprocess(features_list[self.args.valid_attributed_type])
        
        # h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        h0 = torch.zeros(self.all_nodes_num, self.args.att_comp_dim, device=device)
        raw_attributed_node_indices = np.where(self.type_mask == self.args.valid_attributed_type)[0]
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
        if 'one-hot' in self.cluster_op_choice:
            # process one_hot_op
            one_hot_h = []
            for i in range(self.all_nodes_type_num):
                if i == self.args.valid_attributed_type:
                    # one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.hidden_dim)).to(device))
                    one_hot_h.append(torch.zeros((self.node_type_split_list[i], self.args.att_comp_dim)).to(device))
                    continue
                dense_h = self.embedding_list[i](self.one_hot_feature_list[i])
                one_hot_h.append(dense_h)
            one_hot_h = torch.cat(one_hot_h, 0)

        # h_attributed = None
        # h0 = F.dropout(h0, p=self.args.dropout)
        
        # h_attributed = F.elu(h0)
        
        if self.args.shared_ops:
            if self.args.use_skip:
                h_attributed = None
                for k in range(self.args.cluster_num):
                    cur_op_name = self.cluster_op_choice[k]
                    op_idx = self._op_name_list.index(cur_op_name)
                    op = self._shared_ops[op_idx]
                    if op is None:
                        cur_k_res = torch.spmm(self.cluster_mask_matrix[k], one_hot_h)
                    else:
                        cur_k_res = torch.spmm(self.cluster_mask_matrix[k], op(self.g, h0))
                    
                    if h_attributed is None:
                        h_attributed = cur_k_res
                    else:
                        h_attributed = torch.add(h_attributed, cur_k_res)
                    # h_attributed = torch.add(h_attributed, cur_k_res)
                h_attributed = F.elu(h_attributed + F.elu(self.res_fc(h_attributed)))
                # h_attributed = h_attributed + self.res_fc(h_attributed)
                h_attributed = torch.add(h_attributed, h0)
                # h_attributed = F.elu(h_attributed + self.res_fc(h_attributed))
                # h_attributed = F.elu(h_attributed + F.elu(self.res_fc(h_attributed)))
                # h_attributed[raw_attributed_node_indices] = h_raw_attributed_transform
            else:
                h_attributed = h0
                for k in range(self.args.cluster_num):
                    cur_op_name = self.cluster_op_choice[k]
                    op_idx = self._op_name_list.index(cur_op_name)
                    op = self._shared_ops[op_idx]
                    if op is None:
                        cur_k_res = torch.spmm(self.cluster_mask_matrix[k], one_hot_h)
                    else:
                        cur_k_res = torch.spmm(self.cluster_mask_matrix[k], op(self.g, h0))
                    
                    # if self.args.use_skip:
                    #     cur_k_res = cur_k_res + self.res_fc(cur_k_res)
                    #     # cur_k_res = F.elu(cur_k_res + self.res_fc(cur_k_res))
                    
                    h_attributed = torch.add(h_attributed, cur_k_res)
        else:
            # h_attributed = h0
            # for k in range(self.args.cluster_num):
            #     cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h0, one_hot_h)
            #     h_attributed = torch.add(h_attributed, cur_k_res)
            h_attributed = h0
            for k in range(self.args.cluster_num):
                if self._ops[k] is None:
                    cur_k_res = torch.spmm(self.cluster_mask_matrix[k], one_hot_h)
                else:
                    cur_k_res = torch.spmm(self.cluster_mask_matrix[k], self._ops[k](self.g, h0))
                
                if self.args.use_skip:
                    cur_k_res = cur_k_res + self.res_fc(cur_k_res)
                    # cur_k_res = F.elu(cur_k_res + self.res_fc(cur_k_res))
                
                h_attributed = torch.add(h_attributed, cur_k_res)
                
        if self.args.usebn:
            h_attributed = self.bn(h_attributed)
        
        if self.args.useTypeLinear:
            _h = h_attributed
            _h_list = torch.split(_h, self.node_type_split_list)

            h_transform = []
            fc_idx = 0
            for i in range(self.all_nodes_type_num):
                # if i == self.valid_attributed_type:
                #     h_transform.append(_h_list[i])
                #     continue
                h_transform.append(self.fc_list[fc_idx](_h_list[i]))
                fc_idx += 1
            h_transform = torch.cat(h_transform, 0)

            if self.args.usedropout:
                h_transform = F.dropout(h_transform, self.args.dropout)

            # if self.args.use_skip:
                # h_transform = F.elu(h_transform + self.res_fc(h_transform))
                # h_transform = h_transform + self.res_fc(h_transform)
            
            # gnn part
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_transform, mini_batch_input)

        else:
            if self.args.usedropout:
                h_attributed = F.dropout(h_attributed, self.args.dropout)
            
            # if self.args.use_skip:
                # h_attributed = F.elu(h_attributed + self.res_fc(h_attributed))
                # h_attributed = h_attributed + self.res_fc(h_attributed)
            
            node_embedding, logits = self.gnn_model_manager.forward_pass(self.gnn_model, h_attributed, mini_batch_input)

        if self.args.dataset == 'IMDB':
            return node_embedding, logits, F.sigmoid(logits)
        else:
            return node_embedding, logits, logits