# Most of this implementation is based on 
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppn.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        actions = actions
        log_probs = super().log_prob(actions)
        log_probs_viewed = log_probs.view(actions.size(0), -1)
        log_probs_sum = log_probs_viewed.sum(-1)
        log_probs_usqd = log_probs_sum.unsqueeze(-1)
        
        # input()
        return log_probs_usqd

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialize_weight):
        super(Categorical, self).__init__()
        if initialize_weight:
            init_ = lambda m: init(m, nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0), gain=0.01)
        else:
            init_ = lambda m: m

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, as_logits=False):
        x = self.linear(x)
        if not as_logits:
            return FixedCategorical(logits=x)
        else:
            return x


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x, with_dist=True):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        if with_dist:
            return FixedNormal(action_mean, action_logstd.exp())
        else:
            return action_mean, action_logstd.exp()


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, initialize_weight):
        super(Bernoulli, self).__init__()

        if initialize_weight:
            init_ = lambda m: init(m, nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0))
        else:
            init_ = lambda m: m

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, as_logits=False):
        x = self.linear(x)
        if not as_logits:
            return FixedBernoulli(logits=x)
        else:
            return x
