from .GAT import GAT
from .GCN import GCN
from .SimpleHGN import simpleHGN
from .MAGNN import MAGNN_nc_mb, MAGNN_nc
from .HGT import HGT

from .model_manager import ModelManager

MODEL_NAME = {
    'gat': GAT,
    'gcn': GCN,
    'simpleHGN': simpleHGN,
}