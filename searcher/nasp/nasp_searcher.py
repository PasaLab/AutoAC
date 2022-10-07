import time
import torch
import numpy as np
import copy
import gc
from torch.utils.tensorboard import SummaryWriter
from fixed_net import FixedNet

from utils import *
from .supernet import Network_Nasp

class NASPSearcher:
    def __init__(self, data_info, idx_info, train_info, gnn_model_manager, args):
    # def __init__(self, args):
        super(NASPSearcher, self).__init__()
        
        self._logger = args.logger
                
        self.data_info = data_info
        self.idx_info = idx_info
        self.train_info = train_info
        
        self.features_list, self.labels, self.g, self.type_mask, self.dl, self.in_dims, self.num_classes = data_info
        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info

        self.args = args
        
        self._supernet = Network_Nasp(data_info, idx_info, train_info, gnn_model_manager, args)
        # self._supernet = Network_Nasp(args)
        self._supernet = self._supernet.cuda()
        self._supernet_scheduler = None
        if args.useSGD:
            self._supernet_optimizer = torch.optim.SGD(self._supernet.parameters(),
                                            args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
            self._supernet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._supernet_optimizer, float(args.epoch * args.inner_epoch), eta_min=args.lr_rate_min)
        elif args.use_adamw:
            self._supernet_optimizer = torch.optim.AdamW(self._supernet.parameters(), weight_decay=args.weight_decay)
            self._supernet_scheduler = torch.optim.lr_scheduler.OneCycleLR(self._supernet_optimizer, total_steps=args.schedule_step, max_lr=1e-3, pct_start=0.05)
            # self._supernet_scheduler = torch.optim.lr_scheduler.OneCycleLR(self._supernet_optimizer, total_steps=args.schedule_step, max_lr=5e-4, pct_start=0.05)
        else:
            self._supernet_optimizer = torch.optim.Adam(self._supernet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        self._arch_optimizer = torch.optim.Adam(self._supernet.arch_parameters(),
                                                lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        # self._earlystop = EarlyStopping_Search(logger=args.logger, patience=args.patience)
        self._earlystop = EarlyStopping_Search(logger=args.logger, patience=args.patience_search)
        
        self._writer = SummaryWriter(f'/root/tf-logs/{self._save_dir_name}') 

        self.node_assign = None
        
    def _save_search_info(self):
        save_path_name = os.path.join('disrete_arch_info', self._save_dir_name + '_' + self.args.dataset + '_repeat' + str(self.args.cur_repeat))
        if not os.path.exists(save_path_name):
            os.makedirs(save_path_name)

        save_path_name = save_path_name + '.npy'
        save_info = self.get_checkpoint_info()
        np.save(save_path_name, save_info)
    
    @property
    def discreate_file_path(self):
        return self._save_dir_name + '_' + self.args.dataset + '_repeat' + str(self.args.cur_repeat)
    
    def _is_save(self, train_loss, val_loss):
        if val_loss < self._bst_val_loss:
            self._bst_val_loss = val_loss
            return True
        return False
    
    def _train_search(self):
        
        if self.args.useSGD or self.args.use_adamw:
        # if self.args.useSGD:
            # self._supernet_scheduler.step()
            lr = self._supernet_scheduler.get_lr()[0]
        else:
            lr = self._supernet_optimizer.state_dict()['param_groups'][0]['lr']

        if self.args.use_minibatch is False:
            self._supernet.train()
            
            # input, target = convert_np2torch(self.features_list, self.labels, self.args)
            
            # arch params update
            self._supernet.step(self.all_input, self.all_target, None, lr, self._arch_optimizer)

            self._supernet.clip()
            
            # supernet weight params update
            self._supernet_optimizer.zero_grad()
            self._supernet.binarization()
            
            # input, target = convert_np2torch(self.features_list, self.labels, self.args, y_idx=self.train_idx)
            
            if self.args.use_dmon:
                h_attribute, node_embedding, _, logits, dmon_loss, assignments = self._supernet(self.all_input, use_dmon=True)
            else:
                h_attribute, node_embedding, _, logits = self._supernet(self.all_input)
            # self._logger.info(f"search logits: {logits}")
            logits_train = logits[self.train_idx].to(device)
            # train_loss = self._criterion(logits_train, self.search_target[self.train_idx])
            if self.args.use_dmon:
                train_loss_classification = self._criterion(logits_train, self.train_target)
                train_loss = train_loss_classification + self.args.dmon_loss_alpha * dmon_loss
            else:
                train_loss = self._criterion(logits_train, self.train_target)
            train_loss.backward()
            self._supernet.restore()
            
            nn.utils.clip_grad_norm_(self._supernet.parameters(), self.args.grad_clip)
            self._supernet_optimizer.step()

            # self._logger.info("=========== after supernet step =============")
            # for name, parameters in self._supernet.named_parameters():
            #     self._logger.info(f"{name}':' {parameters}")

            if self.args.use_dmon:
                return train_loss, train_loss_classification, dmon_loss, node_embedding, assignments, lr
            else:
                return train_loss, node_embedding, lr
        
        else:
            minibatch_data_info = self._supernet.gnn_model_manager.get_graph_info()
            self.adjlists, self.edge_metapath_indices_list = minibatch_data_info
            
            node_embedding, logits = [], []
            _node_embedding = None
            
            train_loss_avg = 0
            train_loss_classification_avg = 0
            train_loss_dmon_avg = 0
            
            train_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.train_idx)
            val_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.val_idx, shuffle=False)
            for step in range(train_idx_generator.num_iterations()):
                _t_start = time.time()
                
                self._supernet.train()
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                
                val_idx_batch = val_idx_generator.next()
                val_idx_batch.sort()
                val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                    self.adjlists, self.edge_metapath_indices_list, val_idx_batch, device, self.args.neighbor_samples)

                # input, target = convert_np2torch(self.features_list, self.labels, self.args)
                # arch params update
                
                _node_embedding = self._supernet.step(self.all_input, self.all_target, (val_g_list, val_indices_list, val_idx_batch_mapped_list, val_idx_batch), lr, self._arch_optimizer, _node_embedding)
                self._supernet.clip()
                
                # supernet weight params update
                self._supernet_optimizer.zero_grad()
                self._supernet.binarization()

                # input, target = convert_np2torch(self.features_list, self.labels, self.args)
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    self.adjlists, self.edge_metapath_indices_list, train_idx_batch, device, self.args.neighbor_samples)

                if self.args.use_dmon:
                    h_attribute, node_embedding, _, logits, dmon_loss, assignments = self._supernet(self.all_input, (train_g_list, train_indices_list, train_idx_batch_mapped_list, train_idx_batch), use_dmon=True)
                else:
                    h_attribute, node_embedding, _, logits = self._supernet(self.all_input, (train_g_list, train_indices_list, train_idx_batch_mapped_list, train_idx_batch))
                
                _node_embedding = scatter_embbeding(_node_embedding, h_attribute, node_embedding, train_idx_batch)
                
                logits_train = logits.to(device)
                # logits_train = logits[train_idx_batch].to(device)
                # train_loss = self._criterion(logits_train, self.search_target[train_idx_batch])
                if self.args.use_dmon:
                    train_loss_classification = self._criterion(logits_train, self.all_target[train_idx_batch])
                    train_loss = train_loss_classification + self.args.dmon_loss_alpha * dmon_loss
                else:
                    train_loss = self._criterion(logits_train, self.all_target[train_idx_batch])
                train_loss.backward()
                self._supernet.restore()
                
                nn.utils.clip_grad_norm_(self._supernet.parameters(), self.args.grad_clip)
                self._supernet_optimizer.step()

                train_loss_avg += train_loss.item()
                
                if self.args.use_dmon:
                    train_loss_classification_avg += train_loss_classification.item()
                    train_loss_dmon_avg += dmon_loss.item()

                if self.args.use_dmon:
                    self._logger.info('Epoch_batch_{:05d} | lr {:.5f} | Train_Loss {:.4f}| Classification_Loss {:.4f}| Dmon_Loss {:.4f}|Time(s) {:.4f}'.format(
                                    step, lr, train_loss.item(), train_loss_classification.item(), dmon_loss.item(), time.time() - _t_start))
                else:
                    self._logger.info('Epoch_batch_{:05d} | lr {:.5f} | Train_Loss {:.4f}| Time(s) {:.4f}'.format(
                                    step, lr, train_loss.item(), time.time() - _t_start))
                
            train_loss_avg /= train_idx_generator.num_iterations()
            if self.args.use_dmon:
                train_loss_classification_avg /= train_idx_generator.num_iterations()
                train_loss_dmon_avg /= train_idx_generator.num_iterations()
                return train_loss_avg, train_loss_classification_avg, train_loss_dmon_avg, _node_embedding, assignments, lr
            
            return train_loss_avg, _node_embedding, lr
        
    
    def _infer(self):
        self._supernet.eval()
        with torch.no_grad():
            self._supernet.binarization()
            # input, target = convert_np2torch(self.features_list, self.labels, self.args, y_idx=self.val_idx)
            if self.args.use_minibatch is False:
                _, _, _, logits = self._supernet(self.all_input)
                # self._logger.info(f"val logits: {logits}")
                logits_val = logits[self.val_idx].to(device)
            else:
                logits_val = []
                val_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.val_idx, shuffle=False)
                for iteration in range(val_idx_generator.num_iterations()):
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        self.adjlists, self.edge_metapath_indices_list, val_idx_batch, device, self.args.neighbor_samples)
                    _, node_embedding, _, logits = self._supernet(self.all_input, (val_g_list, val_indices_list, val_idx_batch_mapped_list, val_idx_batch))
                    logits_val.append(logits)
                    # logits_val.append(logits[val_idx_batch])
                logits_val = torch.cat(logits_val, 0).to(device)
            
            val_loss = self._criterion(logits_val, self.val_target)
            self._supernet.restore()
        return val_loss.item()

    def search(self):
        self._bst_val_loss = np.inf
        prev_centers = None

        # 格式转换的没什么好说
        self.all_input, self.all_target = convert_np2torch(self.features_list, self.labels, self.args)
        # self._logger.info(f"self.all_input: {self.all_input}\nself.all_target: {self.all_target}")
        
        self.train_input, self.train_target = convert_np2torch(self.features_list, self.labels, self.args, y_idx=self.train_idx)
        self.val_input, self.val_target = convert_np2torch(self.features_list, self.labels, self.args, y_idx=self.val_idx)
        
        for epoch in range(self.args.search_epoch):
            t_start = time.time()
            # print alphas info
            self._logger.info(f"Epoch: {epoch}\n{self._print_alpha_info}")
            
            if self.args.use_dmon:
                train_loss, train_loss_classification, dmon_loss, node_embedding, assignments, lr = self._train_search()
            else:
                train_loss, node_embedding, lr = self._train_search()
            
            if self.args.use_adamw:
                self._supernet_scheduler.step(epoch + 1)
            
            t_train_search = time.time()
            
            val_loss = self._infer()
    
            t_end = time.time()
            
            if self._is_save(train_loss, val_loss):
                self._save_search_info()
            
            if self.args.use_dmon:
                self._logger.info('Epoch {:05d} | lr {:.5f} | Train_Loss {:.4f} | Train_Classification_Loss {:.4f} | Dmon_Loss {:.4f} | Val_Loss {:.4f} | Search Time(s) {:.4f} | Infer Time(s) {:.4f} | Time(s) {:.4f} '.format(
                epoch, lr, train_loss, train_loss_classification, dmon_loss, val_loss, t_train_search - t_start, t_end - t_train_search, t_end - t_start))
            else:
                self._logger.info('Epoch {:05d} | lr {:.5f} | Train_Loss {:.4f} | Val_Loss {:.4f} | Search Time(s) {:.4f} | Infer Time(s) {:.4f} | Time(s) {:.4f} '.format(
                    epoch, lr, train_loss, val_loss, t_train_search - t_start, t_end - t_train_search, t_end - t_start))
            
            if self.args.use_dmon:
                st = time.time()
                assignments = assignments.detach().cpu().numpy()
                # self._logger.info(f"assignments detach time: {time.time() - st}")
                st1 = time.time()
                clusters_info, node_assign = self._supernet.create_new_assignment(assignments)
                # self._logger.info(f"create_new_assignment time: {time.time() - st1}")
                self.node_assign = node_assign
                self._logger.info(f"cluster info:\n{clusters_info[:300]}\n{clusters_info[-300:]}")
                self._writer.add_scalar(f'{self.args.dataset}_Search_train_classification_loss', train_loss_classification, global_step=epoch)
                self._writer.add_scalar(f'{self.args.dataset}_Search_dmon_loss', dmon_loss, global_step=epoch)
            else:
                if (epoch == self.args.warmup_epoch) or (epoch >= self.args.warmup_epoch and ((epoch - self.args.warmup_epoch) % self.args.clusterupdate_round == 0)):
                    last_epoch_centers = copy.deepcopy(prev_centers)
                    for j in range(self.args.cluster_epoch):
                        # execute maximum step to update the cluster center
                        unAttributed_node_emb, new_centers = self._supernet.execute_maximum_step(node_embedding)

                        # check if the iteration ends
                        if prev_centers is not None:
                            is_converge, gap_avg = is_center_close(prev_centers, new_centers, self.args.cluster_eps)
                            self._logger.info(f"Epoch: {epoch} Cluster_epoch: {j} gap_avg: {gap_avg}")
                            if is_converge:
                                self._logger.info(f"cluster center is too close to stop training")
                                break
                            
                        prev_centers = copy.deepcopy(new_centers)
                        clusters_info, node_assign = self._supernet.execute_expectation_step(unAttributed_node_emb, new_centers)

                    self.node_assign = node_assign
                    
                    self._logger.info(f"cluster info:\n{clusters_info[:300]}\n{clusters_info[-300:]}")

                    if last_epoch_centers is not None:
                        is_converge, gap_avg = is_center_close(last_epoch_centers, new_centers, self.args.cluster_eps)
                        self._writer.add_scalar(f'{self.args.dataset}_gap_avg', gap_avg, global_step=epoch)
            
            self._writer.add_scalar(f'{self.args.dataset}_Search_train_loss', train_loss, global_step=epoch)
            self._writer.add_scalar(f'{self.args.dataset}_Search_val_loss', val_loss, global_step=epoch)
            
            self._earlystop(train_loss, val_loss)
            if self._earlystop.early_stop:
                self._logger.info('Eearly stopping!')
                break
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def create_retrain_model(self, alpha, node_assign):
        inner_data_info = self._supernet.gnn_model_manager.get_graph_info()
        gnn_model_manager = self._supernet.gnn_model_manager
        model = FixedNet(self.data_info, self.idx_info, self.train_info, inner_data_info, gnn_model_manager, alpha, node_assign, self.args)

        return model
        
    @property
    def _print_alpha_info(self):
        return self._supernet.print_alpha_params()
    
    @property
    def _save_dir_name(self):
        # create dir
        primitives_str = '.'.join(PRIMITIVES)
        args = self.args
        opt = 'SGD' if args.useSGD else 'Adam'
        # is_rolled = 'unrolled' if args.unrolled else 'one-prox'
        searcher_name = args.searcher_name
        train_info = str(args.warmup_epoch) + str(args.clusterupdate_round) + str(args.cluster_epoch)
        is_use_type_linear = 'typeLinear' if args.useTypeLinear else 'noTypeLinear'
        is_use_dmon = 'useDmon' if args.use_dmon else 'useEM'
        # dir_name = args.dataset + '_' + 'C' + str(args.cluster_num) + '_' + args.gnn_model + \
        #             '_' + opt + \
        #             '_' + primitives_str + '_' + args.time_line + '_' + str(args.seed)
        # self._logger.info(f"train_info type: {type(train_info)}")
        is_shared_ops = 'shared_ops' if args.shared_ops else 'no_shared_ops'
        is_use_5_seeds = 'use5seeds' if args.use_5seeds else 'useSeed123'
        is_use_adamw = 'use_adamw' if args.use_adamw else 'use_adam'
        # is_use_fixed_seeds = 'use_fixseeds' if args.no_use_fixseeds is False else 'no_use_fixseeds'
        patience = 'patience-' + str(args.patience_search) + '_' + str(args.patience_retrain)
        
        is_use_skip = 'use_skip' if args.use_skip else 'not_use_skip'
        
        dir_name = args.gnn_model + '_' + searcher_name + '_' + 'egreedy' + str(args.e_greedy) + \
                    '_' + 'C' + str(args.cluster_num) + '_' + \
                    is_use_dmon + '_coef' + str(args.dmon_loss_alpha) + \
                    '_' + opt + \
                    '_' + primitives_str + '_' + is_use_type_linear + '_' + str(args.warmup_epoch) + str(args.clusterupdate_round) + str(args.cluster_epoch) + \
                    '_' + str(args.att_comp_dim) + '_' + is_shared_ops + \
                    '_' + 'lr' + str(args.lr) + \
                    '_' + 'wd' + str(args.weight_decay) + \
                    '_' + is_use_5_seeds + \
                    '_' + is_use_adamw + \
                    '_' + patience + \
                    '_' + is_use_skip
                    
        self.dir_name = dir_name
        self._logger.info(f"save_dir_name: {dir_name}")

        return dir_name
    
    @property
    def _data_info(self):
        return self.data_info
    
    @property
    def _idx_info(self):
        return self.idx_info
    
    @property
    def _train_info(self):
        return self.train_info
    
    @property
    def writer(self):
        return self._writer
    
    def get_checkpoint_info(self):
        save_info = {}
        save_info['node_assign'] = np.array(self.node_assign)
        save_info['arch_params'] = np.array(self._supernet.arch_parameters()[0].data.cpu())
        
        return save_info
