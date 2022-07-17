import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from dgl.nn.pytorch import GraphConv, APPNPConv, ChebConv
import dgl.function as fn

PRIMITIVES = [
    'gcn',
    'ppnp',
    # 'cheb',
    # 'max',
    'mean',
    
    # 'gcn_1',
    # 'ppnp_1',
    # # 'cheb',
    # # 'max_1',
    # 'mean_1',
    
    # 'gcn_2',
    # 'ppnp_2',
    # # 'cheb',
    # # 'max_2',
    # 'mean_2',
    
    # 'gcn_3',
    # 'ppnp_3',
    # # 'cheb',
    # # 'max_3',
    # 'mean_3',
    
    'one-hot',
]

OPS = {
    'gcn': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'gcn', args),
    'ppnp': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'ppnp', args),
    'max': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'max', args),
    'mean': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'mean', args),
    'cheb': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'cheb', args),
    
    'gcn_1': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'gcn_1', args),
    'ppnp_1': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'ppnp_1', args),
    'max_1': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'max_1', args),
    'mean_1': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'mean_1', args),
    
    'gcn_2': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'gcn_2', args),
    'ppnp_2': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'ppnp_2', args),
    'max_2': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'max_2', args),
    'mean_2': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'mean_2', args),
    
    
    'gcn_3': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'gcn_3', args),
    'ppnp_3': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'ppnp_3', args),
    'max_3': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'max_3', args),
    'mean_3': lambda valid_type, in_dim, out_dim, args: AttributeAggregate(valid_type, in_dim, out_dim, 'mean_3', args),
    
}


class AttributeAggregate(nn.Module):
    def __init__(self, valid_type, in_dim, out_dim, aggr, args):
        super(AttributeAggregate, self).__init__()

        self.valid_type = valid_type
        self.aggr = aggr
        self.args = args
        if aggr == 'gcn':
            # self._op_1 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op_2 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=None, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=False)
            # self._op = GraphConv(in_dim, out_dim, activation=None, weight=False)
        elif aggr == 'ppnp':
            self._ppnp_trans = nn.Linear(in_dim, out_dim, bias=True)
            self._op = APPNPConv(k=10, alpha=0.1)
            # self._op = F.elu(APPNPConv(k=10, alpha=0.1))
        elif aggr == 'cheb':
            self._op = ChebConv(in_dim, out_dim, k=2, bias=True)
        elif aggr in ['max', 'mean']:
            self.fc_neigh = nn.Linear(in_dim, out_dim, bias=True)
            # pass
        elif aggr == 'gcn_1':
            self._op = GraphConv(in_dim, out_dim, activation=None, weight=False)
        elif aggr == 'ppnp_1':
            # self._ppnp_trans = nn.Linear(in_dim, out_dim, bias=True)
            self._op = APPNPConv(k=10, alpha=0.1)
            # self._op = F.elu(APPNPConv(k=10, alpha=0.1))
        elif aggr in ['max_1', 'mean_1']:
            # self.fc_neigh = nn.Linear(in_dim, out_dim, bias=True)
            pass
        elif aggr == 'gcn_2':
            # self._op_1 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op_2 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            self._op = GraphConv(in_dim, out_dim, activation=None, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=False)
            # self._op = GraphConv(in_dim, out_dim, activation=None, weight=False)
        elif aggr == 'ppnp_2':
            self._ppnp_trans = nn.Linear(in_dim, out_dim, bias=True)
            self._op = APPNPConv(k=10, alpha=0.1)
            # self._op = F.elu(APPNPConv(k=10, alpha=0.1))
        elif aggr in ['max_2', 'mean_2']:
            self.fc_neigh = nn.Linear(in_dim, out_dim, bias=True)
            # pass
            
        elif aggr == 'gcn_3':
            # self._op_1 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op_2 = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=None, weight=True)
            # self._op = GraphConv(in_dim, out_dim, activation=F.elu, weight=False)
            # self._op = GraphConv(in_dim, out_dim, activation=None, weight=False)
            self._linear = nn.Linear(in_dim, out_dim, bias=True)
        elif aggr == 'ppnp_3':
            self._ppnp_trans = nn.Linear(in_dim, out_dim, bias=True)
            self._op = APPNPConv(k=10, alpha=0.1)
            self._linear = nn.Linear(out_dim, out_dim, bias=True)
            # self._op = F.elu(APPNPConv(k=10, alpha=0.1))
        elif aggr in ['max_3', 'mean_3']:
            self.fc_neigh = nn.Linear(in_dim, out_dim, bias=True)
            self._linear = nn.Linear(out_dim, out_dim, bias=True)
            # pass
    def forward(self, g, x):
        # if self.args.usedropout:
        #     x = F.dropout(x, self.args.dropout)
        graph = g.local_var()
        if self.aggr in ['gcn', 'cheb']:
            # return self._op_2(g, self._op_1(g, x))
            # return F.elu(self._op(g, x))
            return self._op(g, x)
            # return self._op(g, x) #.flatten(1)? in HGB
        elif self.aggr == 'ppnp':
            h_trans = F.elu(self._ppnp_trans(x))
            return self._op(g, h_trans)
            # return self._op(g, x)
            # return F.elu(self._op(g, x))
            # return F.relu(self._op(g, x))
        elif self.aggr == 'max':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = F.elu(self.fc_neigh(h_neigh))
            return h_neigh
        elif self.aggr == 'mean':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = F.elu(self.fc_neigh(h_neigh))
            # h_neigh = F.elu(h_neigh)
            return h_neigh

        elif self.aggr == 'gcn_1':
            return self._op(g, x)
            # return self._op(g, x) #.flatten(1)? in HGB
        elif self.aggr == 'ppnp_1':
            return self._op(g, x)
        elif self.aggr == 'max_1':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            # h_neigh = F.elu(self.fc_neigh(h_neigh))
            return h_neigh
        elif self.aggr == 'mean_1':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            return h_neigh
        
        elif self.aggr in ['gcn_2']:
            return self._op(g, x)
            # return self._op(g, x) #.flatten(1)? in HGB
        elif self.aggr == 'ppnp_2':
            h_trans = self._ppnp_trans(x)
            return self._op(g, h_trans)
        elif self.aggr == 'max_2':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = self.fc_neigh(h_neigh)
            return h_neigh
        elif self.aggr == 'mean_2':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = self.fc_neigh(h_neigh)
            # h_neigh = F.elu(h_neigh)
            return h_neigh
        
        elif self.aggr in ['gcn_3']:
            # return self._op_2(g, self._op_1(g, x))
            # return F.elu(self._op(g, x))
            # return self._op(g, x)
            conv_res = F.elu(self._op(g, x))
            return conv_res + F.elu(self._linear(x))
            # return F.elu(conv_res + self._linear(conv_res))
            # return self._op(g, x) #.flatten(1)? in HGB
        elif self.aggr == 'ppnp_3':
            # h_trans = F.elu(self._ppnp_trans(x))
            # return self._op(g, h_trans)
            # return self._op(g, x)
            # return F.elu(self._op(g, x))
            # return F.relu(self._op(g, x))
            # ppnp_res = self._op(g, x)
            ppnp_trans = F.elu(self._ppnp_trans(x))
            # ppnp_trans = self._ppnp_trans(x)
            # ppnp_res = F.elu(self._op(g, ppnp_trans))
            ppnp_res = self._op(g, ppnp_trans)
            
            return ppnp_res + F.elu(self._linear(x))
        
        elif self.aggr == 'max_3':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = F.elu(self.fc_neigh(h_neigh))
            # h_neigh = self.fc_neigh(h_neigh)
            return h_neigh + F.elu(self._linear(x))
            # return F.elu(h_neigh + self._linear(h_neigh))
        
        elif self.aggr == 'mean_3':
            graph.srcdata['h'] = x
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            h_neigh = F.elu(self.fc_neigh(h_neigh))
            # h_neigh = self.fc_neigh(h_neigh)
            # h_neigh = F.elu(h_neigh)
            # return h_neigh
            # return F.elu(h_neigh + self._linear(h_neigh))
            return h_neigh + F.elu(self._linear(x))
