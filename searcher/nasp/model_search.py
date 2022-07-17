import copy
import numpy as np
import random
from collections import defaultdict
from sklearn import preprocessing

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
            # print(f"op shape: {op.shape}")
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


class Network_Nasp(nn.Module):
    def __init__(self, g, criterion, train_val_test, type_mask, dl, in_dims, num_classes, args, e_feat=None):
        super(Network_Nasp, self).__init__()
        # graph info
        self.g = g
        self._criterion = criterion
        self.dl = dl
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
        self.type_mask = type_mask

        self.args = args

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

        # random.shuffle(temp_unAttributed_node_id_list)

        self.clusters = []
        # unAttributed node range from (0, unAttributed_nodes_num - 1)
        self.node_cluster_class = [0] * self.unAttributed_nodes_num
        # print(f"node_cluster_class len: {len(self.node_cluster_class)}")

        shift = 0
        for i in range(self.cluster_num):
            if i < self.cluster_num - 1:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, shift + avg_node_num))
                # self.clusters[i]['node_id'] = list(range(shift, shift + avg_node_num))
            else:
                self.clusters.append(defaultdict())
                self.clusters[-1]['node_id'] = list(range(shift, self.unAttributed_nodes_num))
                # self.clusters[i]['node_id'] = list(range(shift, self.unAttributed_nodes_num))
            # self.clusters[i]['node_id'].sort()
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
            # self.cluster_mask_matrix.append(list_to_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num)))
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        # self.alphas = Variable(1e-3 * torch.randn(self.cluster_num, num_ops).cuda(), requires_grad=True)
        self.alphas = Variable(torch.ones(self.cluster_num, num_ops).cuda() / 2, requires_grad=True)
        self._arch_parameters = [self.alphas]
        
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
                # self.embedding_list.append(nn.Linear(dim, hidden_dim, bias=False))
                nn.init.xavier_normal_(self.embedding_list[-1].weight, gain=1.414)

        if self.args.useTypeLinear:
            self.fc_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True) for i in range(self.all_nodes_type_num) if i != self.valid_attr_node_type])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)

        self._ops = nn.ModuleList()
        for k in range(self.cluster_num):
            op = MixedOp(self.valid_attr_node_type, self.g, hidden_dim, hidden_dim, self.args)
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

    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, x, y, is_valid=True):
        # if self.args.dataset == 'IMDB':
        #     node_embedding, _, logits = self(x)
        # else:
        #     node_embedding, _, logits = self(x)
        node_embedding, _, logits = self(x)
        if is_valid:
            input = logits[self.val_idx].cuda()
            target = y[self.val_idx].cuda()
        else:
            input = logits[self.train_idx].cuda()
            target = y[self.train_idx].cuda()  
        # logger.info(f"self._criterion: {self._criterion}")       
        return self._criterion(input, target)

    def execute_maximum_step(self, node_embedding):
        # node_emb = node_embedding.item()
        node_emb = node_embedding.detach().cpu().numpy()

        # print(f"node_emb type: {type(node_emb)}; node_emb shape: {node_emb.shape}")
        assert node_emb.shape[0] == self.all_nodes_num #and node_emb.shape[1] == 

        # print(f"self.originId2clusternodeId:\n{self.originId2clusternodeId}")
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
            # info_str += str(original_id) + '\t' + str(node_cluster_class[i]) + ';\t'
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
            # self.cluster_mask_matrix.append(list_to_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num)))
            self.cluster_mask_matrix.append(to_torch_sp_mat(cur_cluster_node_id, (self.all_nodes_num, self.all_nodes_num), device))
        return self.clusters

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
            m,n = self._arch_parameters[index].size()
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

    def forward(self, features_list):
        # features attribute comletion learning

        h_raw_attributed_transform = self.preprocess(features_list[self.valid_attr_node_type])
        # h_raw_attributed_transform = F.elu(h_raw_attributed_transform)

        # h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device, requires_grad=True)
        h0 = torch.zeros(self.all_nodes_num, self.args.hidden_dim, device=device)
        raw_attributed_node_indices = np.where(self.type_mask == self.valid_attr_node_type)[0]
        h0[raw_attributed_node_indices] = h_raw_attributed_transform

        # logger.info(f"h0.shape: {h0.shape}\nh0:\n{h0}")
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

        # self.alphas_weight = F.softmax(self.alphas, dim=-1)

        # h_attributed = None
        # h_attributed = F.elu(h0)
        h_attributed = None
        for k in range(self.cluster_num):
            cur_k_res = self._ops[k](self.cluster_mask_matrix[k], h0, one_hot_h, self._arch_parameters[0][k])
            # logger.info(f"k: {k} cur_k_res: {cur_k_res}")
            # h_attributed = torch.add(h_attributed, cur_k_res)
            if h_attributed is None:
                h_attributed = cur_k_res
            else:
                h_attributed = torch.add(h_attributed, cur_k_res)
        # h_attributed = torch.add(h_attributed, F.elu(h0))
        h_attributed = torch.add(h_attributed, h0)

        # logger.info(f"h_attributed.shape: {h_attributed.shape}\nh_attributed\n{h_attributed}")

        # logger.info(f"self.e_feat: {self.e_feat}")
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
    
    def genotype(self):
        def _parse(arch_weights):
            gene = []
            arch_indices = torch.argmax(arch_weights, dim=-1)
            for k in arch_indices:
                gene.append(PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu())