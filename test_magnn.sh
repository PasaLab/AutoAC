
# ### use dmon
# # test all settings with seed 123 use dmon; in shared ops
# # shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --shared_ops --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon


# # test all settings with seed 123 use dmon; in shared ops; dmon 0.1
# # # shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1


# # # test all settings with seed 123 use dmon; in shared ops; dmon 0.2
# # # shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2






# # test all settings with seed 123 in shared ops; use dmon; lr 5e-3; weight_decay 1e-3
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon


# # test all settings with seed 123 in no shared ops; use dmon
# # # shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon


# # test all settings with seed 123 in no shared ops; use dmon; lr 5e-3; weight_decay 1e-3
# # #  shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 --use_dmon

# ##  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 --use_dmon






# #### use EM; use 5seeds;
# # test all settings with seed 123 use dmon; in shared ops
# # #  shared ops; darts; 011
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 


# # test all settings with seed 123 use dmon; in shared ops
# # #  shared ops; darts; 15 5 20
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 

# ###  shared ops; nasp; 15 5 20; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 

# ###  shared ops; nasp; 15 5 20; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 



# # test all settings with seed 123 in shared ops; use dmon; lr 5e-3; weight_decay 1e-3
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --lr 5e-3 --weight-decay 1e-3 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 


# # test all settings with seed 123 in no shared ops; use dmon
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 


# # test all settings with seed 123 in no shared ops; use dmon; lr 5e-3; weight_decay 1e-3
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --lr 5e-3 --weight-decay 1e-3 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --lr 5e-3 --weight-decay 1e-3 




# #### use 5 seeds


# # ### use dmon
# # # test all settings with seed 123 use dmon; in shared ops; dmon_coef 0.1
# # ##  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.1

# # ##  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds




# # test all settings with seed 123 use dmon; in shared ops; dmon_coef 0.2
# ##  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.2



# # # test all settings with seed 123 use dmon; in shared ops; dmon_coef 0.3
# # ##  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.3
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.3
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --dmon_loss_alpha 0.3











# ##### search_patience 8_8
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds




# ### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds




# ### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds








# #### test rest cases
# #### dmon coe 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds







# ### test skip-connection
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip






# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip







# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 8 --use_skip












# # ##### search_patience 8_8 imdb 30_30
# # #### dmon coe 0.3
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds




# # #### dmon coe 0.2
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds




# # #### dmon coe 0.1
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds




# # ### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds




# # ### dmon coe 0.5
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds






# # ##### search_patience 8_8 imdb 10_10
# # #### dmon coe 0.3
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds




# # #### dmon coe 0.2
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds




# # #### dmon coe 0.1
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds




# # ### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds




# # ### dmon coe 0.5
# # # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds










#### use EM; use 5seeds;
# test all settings with seed 123 use dmon; in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 

###  shared ops; nasp; 011; e-greedy 0
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds 
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds 
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 


# test all settings with seed 123 use dmon; in shared ops
# #  shared ops; darts; 15 5 20
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 

###  shared ops; nasp; 2_1_20; e-greedy 0
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 2 --clusterupdate-round 1 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds 
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 2 --clusterupdate-round 1 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds 
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds 

# ###  shared ops; nasp; 15 5 20; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 2 --clusterupdate-round 1 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 2 --clusterupdate-round 1 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 



# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4
