import torch
import dgl
import torch.nn as nn
import numpy as np

from .data_process import *
from . import *
from utils.tools import *


class ModelManager:
    def __init__(self, data_info, idx_info, args):
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        
        self.args = args
        
        self.gnn_model_name = args.gnn_model
        
        self._inner_data_info = None
        self._data_process()
        
    def _data_process(self):
        if self.gnn_model_name in ['gat', 'gcn']:
            # data_info, idx_info, train_info = process_gcn_gat(self.args)
            # self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
            pass
        elif self.gnn_model_name in ['simpleHGN', 'magnn']:
            
            if self.gnn_model_name == 'simpleHGN':
                self.e_feat = self._inner_data_info = process_simplehgnn(self.dl, self.g, self.args)
                
            elif self.gnn_model_name == 'magnn':
                self._inner_data_info = data_info = process_magnn(self.args)
                if self.args.dataset == 'IMDB':
                    self.g_lists, self.edge_metapath_indices_lists = self._inner_data_info
                elif self.args.dataset in ['DBLP', 'ACM']:
                    self.adjlists, self.edge_metapath_indices_list = self._inner_data_info
                # self.g_lists, self.edge_metapath_indices_lists = data_info
        elif self.gnn_model_name == 'hgt':
            self.G = process_hgt(self.dl, self.g, self.args)
            
    def create_model_class(self):
        model_name = self.gnn_model_name
        self.heads = [self.args.num_heads] * self.args.num_layers + [1]
        
        if model_name == 'gat':
            model = GAT(self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.heads,
                        F.elu, self.args.dropout, self.args.dropout, self.args.slope, False, self.args.l2norm)
        elif model_name == 'gcn':
            model = GCN(self.g, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, F.elu, self.args.dropout)
        elif model_name == 'simpleHGN':
            model = simpleHGN(self.g, self.args.edge_feats, len(self.dl.links['count']) * 2 + 1, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.heads, F.elu, self.args.dropout, self.args.dropout, self.args.slope, True, 0.05)
        elif model_name == 'magnn':
            if self.args.dataset == 'IMDB':
                num_layers = 2
                etypes_lists = [[[0, 1], [2, 3]],
                                [[1, 0], [1, 2, 3, 0]],
                                [[3, 2], [3, 0, 1, 2]]]
                self.target_node_indices = np.where(self.type_mask == 0)[0]
                model = MAGNN_nc(num_layers, [2, 2, 2], 4, etypes_lists, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                        self.args.rnn_type, self.args.dropout)
                
            elif self.args.dataset == 'DBLP':
                num_layers = 1
                etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
                # etypes_list = [[[0, 3], [0, 1, 4, 3], [0, 2, 5, 3]]]
                # self.target_node_indices = np.where(self.type_mask == 1)[0]
                # self.model = MAGNN_nc(num_layers, [3, 3, 3], 6, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                model = MAGNN_nc_mb(3, 6, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim, self.args.rnn_type, self.args.dropout)
            elif self.args.dataset == 'ACM':
                num_layers = 1
                etypes_list = [[2, 3], [4, 5], [0, 2, 3], [0, 4, 5], [1, 2, 3], [1, 4, 5]]
                # etypes_list = [[[2, 3], [4, 5], [0, 2, 3], [0, 4, 5], [1, 2, 3], [1, 4, 5]]]
                # self.target_node_indices = np.where(self.type_mask == 0)[0]
                # self.model = MAGNN_nc(num_layers, [6], 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                # self.model = MAGNN_nc(num_layers, [6, 6, 6, 6, 6, 6], 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim,
                #         self.args.rnn_type, self.args.dropout)
                model = MAGNN_nc_mb(6, 8, etypes_list, self.in_dims, self.args.hidden_dim, self.num_classes, self.args.num_heads, self.args.attn_vec_dim, self.args.rnn_type, self.args.dropout)
        elif model_name == 'hgt':
            in_dims = [self.args.att_comp_dim for _ in range(self.dl.nodes['total'])]
            model = HGT(self.G, n_inps=in_dims, n_hid=self.args.hidden_dim, n_out=self.num_classes, n_layers=self.args.num_layers, n_heads=self.args.num_heads, use_norm = self.args.use_norm)
        # self.model.to(device)
        return model
        
    def forward_pass(self, gnn_model, h, mini_batch_input=None):
        model = gnn_model
        model_name = self.gnn_model_name
        if model_name == 'gat':
            return model(h)
        elif model_name == 'gcn':
            return model(h)
        elif model_name == 'simpleHGN':
            return model(h, self.e_feat)
        elif model_name == 'magnn':
            if self.args.dataset == 'IMDB':
                return model((self.g_lists, h, self.type_mask, self.edge_metapath_indices_lists), self.target_node_indices)
            elif self.args.dataset in ['DBLP', 'ACM']:
                g_list, indices_list, idx_batch_mapped_list, idx_batch = mini_batch_input
                return model((g_list, h, self.type_mask, indices_list, idx_batch_mapped_list))

            # return self.model((self.g_lists, h, self.type_mask, self.edge_metapath_indices_lists), self.target_node_indices)
        elif model_name == 'hgt':
            node_type_split_list = [self.dl.nodes['count'][i] for i in range(len(self.dl.nodes['count']))]
            h_list = torch.split(h, node_type_split_list)
            features_list = [x for x in h_list]
            return model(self.G, '0', features_list)
        
    def get_graph_info(self):
        return self._inner_data_info
        # return self.adjlists, self.edge_metapath_indices_list
