from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from torch import nn
import torch

class Filter(nn.Module):
    def __init__(self, args, obs_shape, use_recurrent):
        super(Filter, self).__init__()
        # 模型包括attention model and rnn/GRU model, 作为单独模块，将上一时刻的状态信息传递给下一时刻，并作为输入信息，输入attention model中，
        # 输出attention model的weight，并作为输出。其中还有mask信息，用于

        self.args = args
        self.use_recurrent = use_recurrent
        self.inputs_agent_dim = self.args.inputs_agent_dim
        self.rnn_agent_out_dim = self.args.rnn_agent_hidden_dim
        self.inputs_inter_dim = self.args.inputs_inter_dim
        self.rnn_inter_out_dim = self.args.rnn_inter_hidden_dim
        self.recurrent_N = self.args.recurrent_N
        self.use_orthogonal = self.args.use_orthogonal
        self.mlp_hidden_dim = self.args.mlp_hidden_dim
        self.mlp_layer_N = self.args.mlp_layer_N

        if self.use_recurrent:
            self.rnn_agent = RNNLayer(self.inputs_agent_dim, self.rnn_agent_out_dim, self.recurrent_N, self.use_orthogonal)
            self.rnn_intercept = RNNLayer(self.inputs_inter_dim, self.rnn_inter_out_dim, self.recurrent_N, self.use_orthogonal)
            self.encoder = MLPBase(self.args.mlp_args, self.args.obs_shape)
        else:
            self.encoder = MLPBase(self.args.mlp_args, self.args.obs_shape)

    def forward(self, obs_agent, obs_intercept, rnn_agent, rnn_intercept, masks):
        '''
        :param obs_agent: currently agent observation
        :param obs_intercept: currently intercept observation
        :param rnn_agent: rnn embedding of the last time agent observation
        :param rnn_intercept: rnn embedding of the last time intercept observation
        '''
        if self.use_recurrent:
            Feat_obs_agent, H_rnn_agent = self.rnn_agent(obs_agent, rnn_agent, masks)
            Feat_obs_intercept, H_rnn_intercept = self.rnn_intercept(obs_intercept, rnn_intercept, masks)
        else:
            # concatenate the two observations (agent and intercept)
            Feat_obs_agent = obs_agent # shape: (batch_size, agent_num, obs_dim)
            Feat_obs_intercept = obs_intercept # shape: (batch_size, intercept_num, obs_dim)

        result_tensor = torch.cat((Feat_obs_agent.repeat_interleave(self.args.intercept_num, dim=0),
                   Feat_obs_intercept.repeat(self.args.agent_num, dim=1)), dim=1)

        x = self.encoder(result_tensor).view(-1, self.args.agent_num, self.args.intercept_num)
        return torch.normalize(x)

class select_buffer():
    def __init__(self, args, obs_shape):
        self.args = args
        self.obs_shape = obs_shape
        self.buffer = []
    def forward(self):
        pass