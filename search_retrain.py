import sys
# sys.path.append('../../')
import time
import argparse
from scipy.sparse.construct import vstack
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
import gc
from collections import Counter

import dgl
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from model import Network_discrete
from retrainer import Retrainer

# from utils.pytorchtools import EarlyStopping
from utils.tools import *
from utils.data import load_data
from utils.data_process import preprocess

from searcher.darts.model_search import Network_Darts
from searcher.darts.architect import Architect_Darts
from searcher.nasp.supernet import Network_Nasp
from searcher.nasp.architect import Architect_Nasp
from models.model_manager import ModelManager
from searcher import *

logger = get_logger()

SEED = 123
SEED_LIST = [123, 666, 1233, 1024, 2022]
# SEED_LIST = [123, 666, 19, 1024, 2022]
# SEED_LIST = [123, 666, 19, 42, 79]
# SEED_LIST = [123, 1233, 19, 42, 79]
# SEED_LIST = [123, 123, 123, 123, 123]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEARCH_LOG_PATH = 'log_output'
RETRAIN_LOG_PATH = 'retrain_log_output'

def get_args():
    ap = argparse.ArgumentParser(description='AutoHGNN testing for the DBLP dataset')
    ap.add_argument('--dataset', type=str, default='DBLP嚄6', help='Dataset name. Default is DBLP.')                    
    ap.add_argument('--feats-type', type=int, default=6,
                    help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others) We need to try this! Or why did we use glove!;' + 
                        '5 - only term features (zero vec for others).' +
                        '6 - only valid node features (zero vec for others)')
    ap.add_argument('--gnn-model', type=str, default='gat', help='The gnn type in downstream task. Default is gat.')                    
    ap.add_argument('--valid-attributed-type', type=int, default=1, help='The node type of valid attributed node (paper). Default is 1.')                    
    ap.add_argument('--cluster-num', type=int, default=10, help='Number of the clusters for attribute aggreation. Default is 10.')
    ap.add_argument('--cluster-eps', type=float, default=1e-5, help='epsilon for cluster end. Default is 1e-5.')

    ap.add_argument('--att_comp_dim', type=int, default=64, help='Dimension of the attribute completion. Default is 64.')
    
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')

    ap.add_argument('--search_epoch', type=int, default=350, help='Number of epochs. Default is 50.')
    ap.add_argument('--retrain_epoch', type=int, default=500, help='Number of epochs. Default is 50.')
    # ap.add_argument('--search_epoch', type=int, default=2, help='Number of epochs. Default is 50.')
    # ap.add_argument('--retrain_epoch', type=int, default=2, help='Number of epochs. Default is 50.')
    ap.add_argument('--inner-epoch', type=int, default=1, help='Number of inner epochs. Default is 1.')
    # ap.add_argument('--inner-epoch', type=int, default=20, help='Number of inner epochs. Default is 1.')

    # ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--patience_search', type=int, default=30, help='Patience. Default is 30.')
    ap.add_argument('--patience_retrain', type=int, default=30, help='Patience. Default is 30.')
    
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--batch-size-test', type=int, default=32, help='Batch size. Default is 8.')

    ap.add_argument('--momentum', type=float, default=0.9, help='momentum')
    ap.add_argument('--lr', type=float, default=5e-4)
    # ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--lr_rate_min', type=float, default=3e-5, help='min learning rate')

    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    # ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    
    ap.add_argument('--network-momentum', type=float, default=0.9, help='momentum')

    # ap.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')    
    # ap.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    ap.add_argument('--arch_learning_rate', type=float, default=5e-3, help='learning rate for arch encoding')
    ap.add_argument('--arch_weight_decay', type=float, default=1e-5, help='weight decay for arch encoding')

    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--cluster-epoch', type=int, default=1, help='Repeat the cluster epoch each iteration. Default is 1.')
    # ap.add_argument('--cluster-epoch', type=int, default=20, help='Repeat the cluster epoch each iteration. Default is 20.')
    ap.add_argument('--save-postfix', default='DBLP', help='Postfix for the saved model and result. Default is DBLP.')
    ap.add_argument('--feats-opt', type=str, default='1011', help='0100 means 1 type nodes use our processed feature')
    ap.add_argument('--cuda', action='store_true', default=False, help='Using GPU or not.')
    ap.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    ap.add_argument('--useSGD', action='store_true', default=False, help='use SGD as supernet optimize')
    ap.add_argument('--useTypeLinear', action='store_true', default=False, help='use each type linear')
    ap.add_argument('--l2norm', action='store_true', default=False, help='use l2 norm in classification linear')
    ap.add_argument('--cluster-norm', action='store_true', default=False, help='use normalization on node embedding')
    ap.add_argument('--usedropout', action='store_true', default=False, help='use dropout')

    ap.add_argument('--is_unrolled', type=str, default='False', help='help unrolled')
    ap.add_argument('--is_use_type_linear', type=str, default='False', help='help useTypeLinear')
    ap.add_argument('--is_use_SGD', type=str, default='False', help='help useSGD')
    ap.add_argument('--is_use_dropout', type=str, default='False', help='help useSGD')
    ap.add_argument('--time_line', type=str, default="*", help='logging time')

    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--warmup-epoch', type=int, default=0)
    ap.add_argument('--clusterupdate-round', type=int, default=1)

    ap.add_argument('--searcher_name', type=str, default='darts')
    
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--neighbor-samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    
    ap.add_argument('--use-minibatch', type=bool, default=False, help='if use mini-batch')
    ap.add_argument('--shared_ops', action='store_true', default=False, help='ops share weights')
    ap.add_argument('--e_greedy', type=float, default=0, help='nasp e_greedy')
    
    ap.add_argument('--usebn', action='store_true', default=False, help='use dropout')
    
    ap.add_argument('--seed', type=int, default=123, help='random seed.')
    
    ap.add_argument('--use_5seeds', action='store_true', default=False, help='is use 5 different seeds')
    ap.add_argument('--no_use_fixseeds', action='store_true', default=False, help='is use fixed seeds')
    
    ap.add_argument('--use_dmon', action='store_true', default=False, help='is use dmon cluster')
    ap.add_argument('--collapse_regularization', type=float, default=0.1, help='dmon collapse_regularization')
    ap.add_argument('--dmon_loss_alpha', type=float, default=0.3, help='dmon collapse_regularization')
    
    
    ap.add_argument('--tau', type=float, default=1.0, help='dmon collapse_regularization')
    
    ap.add_argument('--schedule_step', type=int, default=350)
    ap.add_argument('--schedule_step_retrain', type=int, default=500)
    ap.add_argument('--use_norm', type=bool, default=False)
    
    ap.add_argument('--use_adamw', action='store_true', default=False, help='is use adamw')
    
    ap.add_argument('--use_skip', action='store_true', default=False, help='is use adamw')
    
    args = ap.parse_args()
    
    return args

def set_random_seed(seed, is_cuda):
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if is_cuda:
        # logger.info('Using CUDA')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        # cudnn.benchmark = True

def retrain(searcher, args, cur_repeat):
    # retrain_time_line = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
    # log_root_path = RETRAIN_LOG_PATH
    # log_save_file = os.path.join(log_root_path, args.dataset + '-' + args.gnn_model + '-' + 'retrain' + '-' + args.time_line + '-' + retrain_time_line + '.log')
    # logger = get_logger(log_root_path, log_save_file)
    
    # args.logger = logger
    
    logger = args.logger
    
    logger.info(f"=============== Retrain Stage Starts:")
    # search_res_file_name = searcher._save_dir_name
    search_res_file_name = searcher.discreate_file_path
    dir_path = os.path.join('disrete_arch_info', search_res_file_name + '.npy')
    checkpoint = np.load(dir_path, allow_pickle=True).item()
    alpha = checkpoint['arch_params']
    node_assign = checkpoint['node_assign']
    
    logger.info(f"node_assign_Counter:\n{Counter(node_assign)}")
    
    retrainer = Retrainer(searcher._data_info, searcher._idx_info, searcher._train_info, searcher.writer, searcher._save_dir_name, args)
    
    # for cur_repeat in range(args.repeat):
        # seed = SEED_LIST[cur_repeat]
        # set_random_seed(seed, args.cuda)
        
    logger.info(f"============= repeat round: {cur_repeat}; seed: {args.seed}")
    
    fixed_model = searcher.create_retrain_model(alpha, node_assign)
    fixed_model = retrainer.retrain(fixed_model, cur_repeat)
    retrainer.test(fixed_model, cur_repeat)
    
    del fixed_model
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"############### Retrain Stage Ends! #################")
    
def search(args):
    log_root_path = SEARCH_LOG_PATH
    log_save_file = os.path.join(log_root_path, args.dataset + '-' + args.gnn_model + '-' + 'search' + '-' + args.time_line + '.log')
    logger = get_logger(log_root_path, log_save_file)
    
    args.logger = logger
    
    logger.info(f"=============== Search Args:\n{args}")
    t = time.time()
    # load data
    features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_data(args.dataset)
    logger.info(f"node_type_num: {len(dl.nodes['count'])}")
    # logger.info(f"type_mask ")
    
    # data process
    data_info, idx_info, train_info = preprocess(features_list, adjM, type_mask, labels, train_val_test_idx, dl, args)
    
    logger.info(f"=============== Prepare basic data stage finish, use {time.time() - t} time.")
    
    gnn_model_manager = ModelManager(data_info, idx_info, args)
    # gnn_model.create_model_class()

    # 调用的是NASPSearcher
    searcher = SEARCHER_NAME[args.searcher_name](data_info, idx_info, train_info, gnn_model_manager, args)
    
    searcher.search()
    
    logger.info(f"############### Search Stage Ends! ###############")
    
    return searcher

if __name__ == '__main__':

    args = get_args()
    
    # if args.is_unrolled == 'True':
    #     args.unrolled = True
    
    if args.is_use_type_linear == 'True':
        args.useTypeLinear = True

    if args.is_use_SGD == 'True':
        args.useSGD = True

    if args.is_use_dropout == 'True':
        args.usedropout = True

    if args.dataset in ['ACM', 'IMDB']:
        args.valid_attributed_type = 0
        args.feats_opt = '0111'
    elif args.dataset == 'Freebase':
        args.feats_type = 1
        # args.valid_attributed_type = 4
        # args.feats_opt = '11110111'
        # args.valid_attributed_type = 0
        # args.feats_opt = '01111111'
        args.valid_attributed_type = 1
        args.feats_opt = '10111111'
    
    if args.dataset in ['DBLP', 'ACM'] and args.gnn_model == 'magnn':
        args.use_minibatch = True
    
    if args.gnn_model in ['gcn', 'hgt']:
        args.last_hidden_dim = args.hidden_dim
    elif args.gnn_model in ['gat', 'simpleHGN']:
        args.last_hidden_dim = args.hidden_dim * args.num_heads
    elif args.gnn_model in ['magnn']:
        if args.dataset == 'IMDB':
            args.last_hidden_dim = args.hidden_dim * args.num_heads
        elif args.dataset in ['DBLP', 'ACM']:
            args.last_hidden_dim = args.hidden_dim
        # args.last_hidden_dim = args.attn_vec_dim * args.num_heads
        
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    args.time_line = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    
    if args.use_5seeds:
        # set random seed
        for cur_repeat, seed in enumerate(SEED_LIST):
            
            set_random_seed(seed, args.cuda)

            args.seed = seed
            args.cur_repeat = cur_repeat
            
            searcher = search(args)
        
            retrain(searcher, args, cur_repeat)
            
    elif args.no_use_fixseeds:
        # not fix seeds
        for cur_repeat in range(args.repeat):
            searcher = search(args)
            retrain(searcher, args, cur_repeat)
    else:
        
        set_random_seed(SEED, args.cuda)

        args.seed = SEED
        
        searcher = search(args)
        
        for cur_repeat in range(args.repeat):        
            retrain(searcher, args, cur_repeat)
    