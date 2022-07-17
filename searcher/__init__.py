from .nasp.nasp_searcher import NASPSearcher
from .darts.darts_searcher import DARTSSearcher
from .nasp_all_nodes.nasp_searcher import NASPSearcherAllnodes

SEARCHER_NAME = {
    'nasp': NASPSearcher,
    'darts': DARTSSearcher,
    'nasp_all_nodes': NASPSearcherAllnodes 
}
