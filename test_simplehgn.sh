
# ##### dmon
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --num-layers 5


# # test all settings with seed 123 in no_shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon --num-layers 5


# ##### dmon not use no shared ops acm patience 30
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --num-layers 5


# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5


# ###### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5



# ###### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5


# ###### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5


# ##### dmon not use no shared ops acm patience 8
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --num-layers 5


# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5


# ###### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5


# # ###### EM
# # # test all settings with seed 123 in shared ops
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --num-layers 5 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --num-layers 5 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --num-layers 5 

# ##  shared ops; darts; 15 5 20
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --num-layers 5 

# ###  shared ops; nasp; 15 5 20; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --num-layers 5 

# ###  shared ops; nasp; 15 5 20; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 --num-layers 5 


# # # test all settings with seed 123 in no_shared ops
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 




# ########### use 5 seeds
# ##### dmon 0.3
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --num-layers 5

# ##### dmon 0.1
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5

# ##### dmon 0.2
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5







# ########### use 5 seeds; patience 8 30
# ##### dmon 0.3
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ##### dmon 0.1
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ##### dmon 0.2
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 30





# ########### use 5 seeds; patience 8 8
# ##### dmon 0.3
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ##### dmon 0.1
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ##### dmon 0.2
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 8 --patience_retrain 8
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --num-layers 5 --patience_search 8 --patience_retrain 8





# # test all settings with seed 123 in no_shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_dmon --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_dmon --use_5seeds




# ###### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds


# ###### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds



# ###### EM
# # # test all settings with seed 123 in shared ops
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_5seeds --num-layers 5 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds --num-layers 5 


# ##  shared ops; darts; 15 5 20
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds  
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds --num-layers 5  

# ###  shared ops; nasp; 15 5 20; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds  
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds  
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --use_5seeds --num-layers 5  

# ###  shared ops; nasp; 15 5 20; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds  
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds  
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1 --use_5seeds --num-layers 5  


# # # test all settings with seed 123 in no_shared ops
# ##  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --use_5seeds 

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_5seeds 
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --e_greedy 0.1 --use_5seeds 




# # test no fixed seeds
# ##### dmon
# # test all settings with seed 123 in shared ops
# #  shared ops; darts; 011
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --unrolled --shared_ops --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --shared_ops --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --cluster-num $2 --patience 30 --shared_ops --use_dmon --num-layers 5 --use_5seeds --no_use_fixseeds

# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --use_dmon --num-layers 5 --use_5seeds --no_use_fixseeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --e_greedy 0.1 --use_dmon --use_5seeds --no_use_fixseeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience 30 --shared_ops --e_greedy 0.1 --use_dmon --num-layers 5 --use_5seeds --no_use_fixseeds




###### acm 30
# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5











# ###### acm 8_30
# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5




# # #### dmon coe 0.5
# # # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5



###### acm 8_30
#### dmon coe 0.4
# test all settings with seed 123 in shared ops
###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 8 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5





# ##### search_patience 10 10 search_retrain 10
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5















# ##### search_patience 10 10 search_retrain 15
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 10 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 15 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5








# ##### search_patience 10_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 10 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds













# ##### search_patience 15_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 15 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds












# ##### search_patience 5_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 5 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds














# ##### search_patience 8_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds




# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds










# ##### Use EM; search_patience 30_30; 5seeds
# # test all settings  in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops  --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1  --num-layers 5 --use_5seeds


# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops  --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1  --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 15 --clusterupdate-round 5 --cluster-epoch 20 --unrolled --shared_ops --e_greedy 0.1  --num-layers 5 --use_5seeds




# ### test skip-connection
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip






# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip







# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 4 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 16 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 8 --patience_retrain 30 --num-layers 5+rixRnKNOG --use_skip














# ##### search_patience 30_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds



# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds


# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# # ###  shared ops; nasp; 011; e-greedy 0
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 30 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds



# ### test skip-connection 30_30 wd5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4



# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 5e-4
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 5e-4






# ### test skip-connection 30_30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip



# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip






# ### test skip-connection; use op_1 + use_skip; 30_30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip




### test skip-connection; use op + skip-connection ; 30_30
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip

CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip


CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip



CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip
CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip




# ### test skip-connection; use op + skip-connection + elu; 30_30; wd1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3 --use_skip

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3 --use_skip


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3 --use_skip



# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --use_skip --weight_decay 1e-3 --use_skip
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --use_skip --weight_decay 1e-3 --use_skip







# ### test no skip-connection 30_30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5






# ### test no skip-connection 30_30 wd1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --weight_decay 1e-3
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5 --weight_decay 1e-3




# ### test darts 30_30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5

# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5


# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name darts --unrolled --cluster-num $2 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --patience_search 30 --patience_retrain 30 --num-layers 5






# ##### search_patience 8_30 tau=0.1
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds --tau 0.1






# ##### search_patience 8_30
# #### dmon coe 0.3
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.3 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.2
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.2 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.1
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds

# # ###  shared ops; nasp; 011; e-greedy 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --use_5seeds --tau 0.1
# # CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.1 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.4
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.4 --num-layers 5 --use_5seeds --tau 0.1




# #### dmon coe 0.5
# # test all settings with seed 123 in shared ops
# ###  shared ops; nasp; 011; e-greedy 0
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds --tau 0.1

# ###  shared ops; nasp; 011; e-greedy 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --use_5seeds --tau 0.1
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num $2 --patience_search 8 --patience_retrain 30 --warmup-epoch 0 --clusterupdate-round 1 --cluster-epoch 1 --unrolled --shared_ops --e_greedy 0.1 --use_dmon --dmon_loss_alpha 0.5 --num-layers 5 --use_5seeds --tau 0.1
