
"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""

from torch.nn import Module
from torch.nn import  Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()

        self.learning_rate = 0.01
        self.numTrainingGames = 8000
        self.batch_size = 64

        self.fc1 = Linear(state_dim, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, action_dim)

        self.double()

        self.parameters = [self.fc1.weight, self.fc1.bias,
                           self.fc2.weight, self.fc2.bias,
                           self.fc3.weight, self.fc3.bias]
        self.optimizer = optim.SGD(self.parameters, lr=self.learning_rate, momentum=0.9)


    def get_loss(self, states, Q_target):
        preds = self.forward(states)
        return mse_loss(preds, Q_target)


    def forward(self, states):
        # Do NOT wrap with tensor() when states is already a tensor
        # — that would detach it and break gradient flow.
        if hasattr(states, 'double'):
            x = states.double()
        else:
            x = tensor(states).double()
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        loss = self.get_loss(states, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
