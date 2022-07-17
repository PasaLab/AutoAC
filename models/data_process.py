import torch
import torch.nn as nn
import dgl
import networkx as nx

from utils import *

def process_hgt(dl, g, args):
    edge_dict = {}

    for i, meta_path in dl.links['meta'].items():
        edge_dict[(str(meta_path[0]), str(meta_path[0]) + '_' + str(meta_path[1]), str(meta_path[1]))] = (torch.tensor(dl.links['data'][i].tocoo().row - dl.nodes['shift'][meta_path[0]]), torch.tensor(dl.links['data'][i].tocoo().col - dl.nodes['shift'][meta_path[1]]))

    node_count = {}
    for i, count in dl.nodes['count'].items():
        print(i, node_count)
        node_count[str(i)] = count

    G = dgl.heterograph(edge_dict, num_nodes_dict = node_count, device=device)
    """
    for ntype in G.ntypes:
        G.nodes[ntype].data['inp'] = dataset.nodes['attr'][ntype]
        # print(G.nodes['attr'][ntype].shape)
    """
    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long).to(device) * G.edge_dict[etype] 

    return G

def process_simplehgnn(dl, g, args):
    def process_edge2type():
        edge2type = {}
        if args.dataset == 'IMDB':
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    edge2type[(u, v)] = k
            for i in range(dl.nodes['total']):
                edge2type[(i, i)] = len(dl.links['count'])
        else:
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    edge2type[(u, v)] = k 
            for i in range(dl.nodes['total']):
                if (i, i) not in edge2type:
                    edge2type[(i, i)] = len(dl.links['count'])
            for k in dl.links['data']:
                for u, v in zip(*dl.links['data'][k].nonzero()):
                    if (v, u) not in edge2type:
                        edge2type[(v, u)] = k + 1 + len(dl.links['count'])
        return edge2type
    
    e_feat = None
    edge2type = process_edge2type()
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    # logger.info(f"e_feat: {e_feat}")
    return e_feat


def process_magnn(args):
    logger = args.logger
    dataset_name = args.dataset
    
    def get_adjlist_pkl(dl, meta, type_id=0, return_dic=True, symmetric=False):
        meta010 = dl.get_meta_path(meta).tocoo()
        adjlist00 = [[] for _ in range(dl.nodes['count'][type_id])]
        for i, j, v in zip(meta010.row, meta010.col, meta010.data):
            adjlist00[i - dl.nodes['shift'][type_id]].extend([j - dl.nodes['shift'][type_id]] * int(v))
        adjlist00 = [' '.join(map(str, [i] + sorted(x))) for i, x in enumerate(adjlist00)]
        meta010 = dl.get_full_meta_path(meta, symmetric=symmetric)
        idx00 = {}
        for k in meta010:
            idx00[k] = np.array(sorted([tuple(reversed(i)) for i in meta010[k]]), dtype=np.int32).reshape([-1, len(meta)+1])
        if not return_dic:
            idx00 = np.concatenate(list(idx00.values()), axis=0)
        # logger.info(f"type_id: {type_id}, idx00: {idx00}")
        return adjlist00, idx00

    def load_DBLP_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/DBLP')
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)], symmetric=True)
        logger.info('meta path 2 done')
        adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)], symmetric=True)
        logger.info('meta path 3 done')
        
        # adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], return_dic=False, symmetric=True)
        # G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        # logger.info('meta path 1 done')
        # adjlist01, idx01 = get_adjlist_pkl(dl, [(0,1), (1,2), (2,1), (1,0)], return_dic=False, symmetric=True)
        # G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        # logger.info('meta path 2 done')
        # adjlist02, idx02 = get_adjlist_pkl(dl, [(0,1), (1,3), (3,1), (1,0)], return_dic=False, symmetric=True)
        # G02 = nx.readwrite.adjlist.parse_adjlist(adjlist02, create_using=nx.MultiDiGraph)
        # logger.info('meta path 3 done')
        
        features = []
        for i in range(4):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        # 0 for authors, 1 for papers, 2 for terms, 3 for conferences
        features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(4):
            type_mask[dl.nodes['shift'][i]: dl.nodes['shift'][i] + dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx
        
        # Using PAP to define relations between papers.
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adjM = adjM.toarray()
        adjM = torch.FloatTensor(adjM).to(device)
        a_mask = np.where(type_mask == 0)[0]
        p_mask = np.where(type_mask == 1)[0]
        a_mask = torch.LongTensor(a_mask).to(device)
        p_mask = torch.LongTensor(p_mask).to(device)
        adjM[p_mask, :][:, p_mask] = torch.mm(adjM[p_mask, :][:, a_mask], adjM[a_mask, :][:, p_mask])
        adjM = adjM.data.cpu().numpy()
        torch.cuda.empty_cache()

        return [adjlist00, adjlist01, adjlist02], \
            [idx00, idx01, idx02], \
            [features_0, features_1, features_2, features_3],\
            adjM, \
            type_mask,\
            labels,\
            train_val_test_idx,\
            dl

        # return [[G00, G01, G02]], \
        #     [[idx00, idx01, idx02]], \
        #     [features_0, features_1, features_2, features_3],\
        #     adjM, \
        #     type_mask,\
        #     labels,\
        #     train_val_test_idx,\
        #     dl
            
    def load_ACM_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/ACM')
        dl.get_sub_graph([0, 1, 2])
        #dl.links['data'][0] += sp.eye(dl.nodes['total'])
        for i in range(dl.nodes['count'][0]):
            if dl.links['data'][0][i].sum() == 0:
                dl.links['data'][0][i,i] = 1
            if dl.links['data'][1][i].sum() == 0:
                dl.links['data'][1][i,i] = 1
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], symmetric=True)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], symmetric=True)
        logger.info('meta path 2 done')
        adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)])
        logger.info('meta path 3 done')
        adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)])
        logger.info('meta path 4 done')
        adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)])
        logger.info('meta path 5 done')
        adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)])
        logger.info('meta path 6 done')
        # adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], return_dic=False, symmetric=True)
        # G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        # logger.info('meta path 1 done')
        # adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], return_dic=False, symmetric=True)
        # G01= nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        # logger.info('meta path 2 done')
        # adjlist02, idx02 = get_adjlist_pkl(dl, [0, (0,1), (1,0)], return_dic=False)
        # G02 = nx.readwrite.adjlist.parse_adjlist(adjlist02, create_using=nx.MultiDiGraph)
        # logger.info('meta path 3 done')
        # adjlist03, idx03 = get_adjlist_pkl(dl, [0, (0,2), (2,0)], return_dic=False)
        # G03 = nx.readwrite.adjlist.parse_adjlist(adjlist03, create_using=nx.MultiDiGraph)
        # logger.info('meta path 4 done')
        # adjlist04, idx04 = get_adjlist_pkl(dl, [1, (0,1), (1,0)], return_dic=False)
        # G04 = nx.readwrite.adjlist.parse_adjlist(adjlist04, create_using=nx.MultiDiGraph)
        # logger.info('meta path 5 done')
        # adjlist05, idx05 = get_adjlist_pkl(dl, [1, (0,2), (2,0)], return_dic=False)
        # G05 = nx.readwrite.adjlist.parse_adjlist(adjlist05, create_using=nx.MultiDiGraph)
        # logger.info('meta path 6 done')
        features = []
        types = len(dl.nodes['count'])
        for i in range(types):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        #features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        adjM = adjM.toarray()

        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(types):
            type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx

        return [adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05], \
                [idx00, idx01, idx02, idx03, idx04, idx05], \
                features, \
                adjM, \
                type_mask,\
                labels,\
                train_val_test_idx,\
                dl

        # return [[G00, G01, G02, G03, G04, G05]], \
        #         [[idx00, idx01, idx02, idx03, idx04, idx05]], \
        #         features, \
        #         adjM, \
        #         type_mask,\
        #         labels,\
        #         train_val_test_idx,\
        #         dl
                
    def load_IMDB_data():
        from utils.data_loader import data_loader
        dl = data_loader('data/IMDB')
        adjlist00, idx00 = get_adjlist_pkl(dl, [(0,1), (1,0)], 0, False, True)
        G00 = nx.readwrite.adjlist.parse_adjlist(adjlist00, create_using=nx.MultiDiGraph)
        logger.info('meta path 1 done')
        adjlist01, idx01 = get_adjlist_pkl(dl, [(0,2), (2,0)], 0, False, True)
        G01 = nx.readwrite.adjlist.parse_adjlist(adjlist01, create_using=nx.MultiDiGraph)
        logger.info('meta path 2 done')
        adjlist10, idx10 = get_adjlist_pkl(dl, [(1,0), (0,1)], 1, False, True)
        G10 = nx.readwrite.adjlist.parse_adjlist(adjlist10, create_using=nx.MultiDiGraph)
        logger.info('meta path 3 done')
        adjlist11, idx11 = get_adjlist_pkl(dl, [(1,0), (0,2), (2,0), (0, 1)], 1, False, True)
        G11 = nx.readwrite.adjlist.parse_adjlist(adjlist11, create_using=nx.MultiDiGraph)
        logger.info('meta path 4 done')
        adjlist20, idx20 = get_adjlist_pkl(dl, [(2,0), (0,2)], 2, False, True)
        G20 = nx.readwrite.adjlist.parse_adjlist(adjlist20, create_using=nx.MultiDiGraph)
        logger.info('meta path 5 done')
        adjlist21, idx21 = get_adjlist_pkl(dl, [(2,0), (0,1), (1,0), (0,2)], 2, False, True)
        G21 = nx.readwrite.adjlist.parse_adjlist(adjlist21, create_using=nx.MultiDiGraph)
        logger.info('meta path 6 done')
        features = []
        types = len(dl.nodes['count'])
        for i in range(types):
            th = dl.nodes['attr'][i]
            if th is None:
                features.append(np.eye(dl.nodes['count'][i], dtype=np.float32))
            else:
                if type(th) is np.ndarray:
                    features.append(th)
                else:
                    features.append(th.toarray())
        #features_0, features_1, features_2, features_3 = features

        adjM = sum(dl.links['data'].values())
        type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
        for i in range(types):
            type_mask[dl.nodes['shift'][i]:dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        #labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adjM = adjM.toarray()
        adjM = torch.FloatTensor(adjM).to(device)
        m_mask = np.where(type_mask == 0)[0]
        d_mask = np.where(type_mask == 1)[0]
        a_mask = np.where(type_mask == 2)[0]
        a_mask = torch.LongTensor(a_mask).to(device)
        m_mask = torch.LongTensor(m_mask).to(device)
        d_mask = torch.LongTensor(d_mask).to(device)

        adjM[m_mask, :][:, m_mask] = torch.mm(adjM[m_mask, :][:, a_mask], adjM[a_mask, :][:, m_mask])
        adjM[m_mask, :][:, m_mask] = adjM[m_mask, :][:, m_mask] + torch.mm(adjM[m_mask, :][:, d_mask], adjM[d_mask, :][:, m_mask])
        adjM = adjM.data.cpu().numpy()
        torch.cuda.empty_cache()

        return [[G00, G01], [G10, G11], [G20, G21]], \
                [[idx00, idx01], [idx10, idx11], [idx20, idx21]], \
                features, \
                adjM, \
                type_mask,\
                labels,\
                train_val_test_idx,\
                dl
            
    if dataset_name == 'DBLP':
        adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_DBLP_data()
        return (adjlists, edge_metapath_indices_list)
    elif dataset_name == 'ACM':
        adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_ACM_data()
        return (adjlists, edge_metapath_indices_list)
    elif dataset_name == 'IMDB':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_IMDB_data()
        edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                    edge_metapath_indices_lists]
        g_lists = []
        for nx_G_list in nx_G_lists:
            g_lists.append([])
            for nx_G in nx_G_list:
                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(nx_G.number_of_nodes())
                g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
                g_lists[-1].append(g.to(device))
        return (g_lists, edge_metapath_indices_lists)
    
    # if dataset_name == 'DBLP':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_DBLP_data()
    # elif dataset_name == 'ACM':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_ACM_data()
    # elif dataset_name == 'IMDB':
    #     nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx, dl = load_IMDB_data()
    
            
    # edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
    #                             edge_metapath_indices_lists]
    # g_lists = []
    # for nx_G_list in nx_G_lists:
    #     g_lists.append([])
    #     for nx_G in nx_G_list:
    #         g = dgl.DGLGraph(multigraph=True)
    #         g.add_nodes(nx_G.number_of_nodes())
    #         g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
    #         g_lists[-1].append(g.to(device))
    # return (g_lists, edge_metapath_indices_lists)
