## This model based on this paper: https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.1c05264
## And is coming from this URL 
## https://github.com/zavalab/ML/blob/master/CMC_GCN/saved_models/gnn_logs_save_202_hu256_lr0.005_best_trainalles_seed4592/ep200bs5lr0.005hu256es.pth.tar

# !pip install dgllife
# !pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph, BaseAtomFeaturizer, atomic_number

# Define the GCNReg model as per the repository's implementation
class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g):
        h = g.ndata['h']
        if self.saliency:
            h.requires_grad = True
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        x = F.relu(self.classify1(hg))
        x = F.relu(self.classify2(x))
        x = self.classify3(x)
        return x

from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

def get_graph_from_smiles(smiles):
    graph = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer())
    return graph

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def predict_single_point(smiles, model_path, device):
    # Load the model
    model = GCNReg(in_dim=74, hidden_dim=256, n_classes=1)  # Adjust input and hidden dimensions as needed
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare the input
    graph = get_graph_from_smiles(smiles)
    graph = graph.to(device)
    graph.ndata['h'] = graph.ndata['h'].to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(graph)
    
    return output.cpu().item()