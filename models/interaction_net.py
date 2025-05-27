import torch
import torch_geometric as pyg
from torch import nn

from ..utils import make_mlp

class InteractionNet(pyg.nn.MessagePassing):
    def __init__(
            self,
            edge_index,
            input_dim,
            update_edges=True,
            hidden_layers=1,
            hidden_dim=None,
            aggr="sum",
    ):
        
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__()
        self.aggr = aggr

        if hidden_dim is None:
            hidden_dim = input_dim

        ####### this should be checked
        # indice start at 0 
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # store num of receiver nodes
        self.num_rec = edge_index[1].max() + 1
        # make sender indices after receiver
        edge_index[0] = (edge_index[0] + self.num_rec)
        #######

        self.register_buffer("edge_index", edge_index, persistent=False)

        edge_mlp_dim = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1) # 3: r, s, e
        rec_mlp_dim = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1) # 2: r, aggr_e
        self.edge_mlp = make_mlp(edge_mlp_dim)
        self.rec_mlp = make_mlp(rec_mlp_dim)

        self.update_edges = update_edges

    def forward(self, send_feature, rec_feature, edge_feature):
        send_rec_feature = torch.cat((send_feature, rec_feature), dim=-2)
        edge_feature_aggr, edge_feature_update = self.propagate(
            self.edge_index, x=send_rec_feature, edge_attr=edge_feature
        )
        rec_feature_update = self.rec_mlp(torch.cat(rec_feature, edge_feature_aggr),dim=-1)

        rec_feature = rec_feature + rec_feature_update

        if self.update_edges:
            edge_feature = edge_feature + edge_feature_update
            return rec_feature, edge_feature
        
        return rec_feature
    
    def message(self, x_j, x_i, edge_attr):
        """
        x_j: sender node features
        x_i: receiver node features
        edge_attr: edge features
        """
        return self.edge_mlp(torch.cat((x_j, x_i, edge_attr), dim=-1))
    
    def aggregate(self, inputs, index, ptr, dim_size):

        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs

