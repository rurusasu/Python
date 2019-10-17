# cording : utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe


tfe.enable_eager_execution()


# PyTorchのNetwork定義
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Liner(state_dim, 400)
        self.l2 = nn.Liner(400, 300)
        self.l3 = nn.Liner(300, action_dim)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(tf.keras.Model):


