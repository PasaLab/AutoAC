
# simplehgn
# python3 search_retrain.py --dataset=DBLP --gnn-model=simpleHGN --searcher_name nasp --cluster-num 8 --patience_search 8 --patience_retrain 30 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
 python3 search_retrain.py --dataset=ACM --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --patience_search 8 --patience_retrain 30 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=simpleHGN --searcher_name nasp --cluster-num 12 --patience_search 8 --patience_retrain 30 --shared_ops --use_dmon --dmon_loss_alpha 0.4 --use_5seeds


# # magnn
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=DBLP --gnn-model=magnn --searcher_name nasp --cluster-num 4 --patience_search 8 --patience_retrain 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --e_greedy 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=ACM --gnn-model=magnn --searcher_name nasp --cluster-num 4 --patience_search 8 --patience_retrain 8 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --e_greedy 0.1 --use_5seeds
# CUDA_VISIBLE_DEVICES=$1 python3 search_retrain.py --dataset=IMDB --gnn-model=magnn --searcher_name nasp --cluster-num 16 --patience_search 30 --patience_retrain 30 --shared_ops --use_dmon --dmon_loss_alpha 0.5 --e_greedy 0.1 --use_5seeds
