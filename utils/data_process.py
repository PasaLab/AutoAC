import torch
import numpy as np
import dgl
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def preprocess(features_list, adjM, type_mask, labels, train_val_test_idx, dl, args):
    if args.feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif args.feats_type == 6:
        # valid指的是有属性的
        save = args.valid_attributed_type
        feature_dim = features_list[save].shape[1]
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(feature_dim)
            else:
                in_dims.append(feature_dim)
                features_list[i] = np.zeros((features_list[i].shape[0], feature_dim))
                # features_list[i] = torch.zeros((features_list[i].shape[0], feature_dim)).to(device)
    elif args.feats_type == 1:
        save = args.valid_attributed_type
        feature_dim = features_list[save].shape[0]
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                indices = np.vstack((np.arange(feature_dim), np.arange(feature_dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(feature_dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([feature_dim, feature_dim])).to(device)
                # features_list[i] = np.identity(feature_dim)
                in_dims.append(feature_dim)
            else:
                in_dims.append(feature_dim)
                features_list[i] = np.zeros((features_list[i].shape[0], feature_dim))
                # features_list[i] = torch.zeros((features_list[i].shape[0], feature_dim)).to(device)

    in_dims = [features.shape[1] for features in features_list]

    # labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    if args.dataset == 'IMDB':
        g = dgl.DGLGraph(adjM)
    else:
        g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    if args.dataset == 'IMDB':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    
    num_classes = dl.labels_train['num_classes']
    
    return (features_list, labels, g, type_mask, dl, in_dims, num_classes), (train_idx, val_idx, test_idx), criterion
