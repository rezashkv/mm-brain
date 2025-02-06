import torch
import torch.nn as nn
from .single_modal_models import MLP, FTTransformer


class MultiModalTeacher(nn.Module):
    """A MultiModal Model combining two MLPs each acting on one modality

        The following scheme describes the architecture:

        .. code-block:: text

            t1_net: MLP
            snp_net: MLP
            regressor: Final regression head
        """
    def __init__(self,
                 t1_net_type: str,
                 t1_net_params: dict,
                 snp_net_type: str,
                 snp_net_params: dict,
                 output_dim: int,
                 task: str):
        super().__init__()

        if t1_net_type == 'mlp':
            self.t1_net = MLP.make_baseline(**t1_net_params)
        else:
            self.t1_net = FTTransformer.make_default(**t1_net_params)
            self.t1_net.d_in = t1_net_params['n_num_features']
            self.t1_net.d_out = t1_net_params['d_out']
        if snp_net_type == 'mlp':
            self.snp_net = MLP.make_baseline(**snp_net_params)
        else:
            self.snp_net = FTTransformer.make_default(**snp_net_params)
            self.snp_net.d_in = snp_net_params['n_num_features']
            self.snp_net.d_out = snp_net_params['d_out']
        self.regressor = nn.Linear(self.t1_net.d_out + self.snp_net.d_out, output_dim)
        self.task = task
        if self.task == 'classification':
            self.clf = nn.LogSoftmax(dim=1)


    def forward(self, t1_data, snp_data):
        t1_features = self.t1_net(t1_data)
        snp_features = self.snp_net(snp_data)
        features = torch.cat([t1_features, snp_features], dim=1)
        if self.task == 'regression':
            return self.regressor(features), t1_features, snp_features
        return self.clf(self.regressor(features)), t1_features, snp_features
