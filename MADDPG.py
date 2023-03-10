from model import MADDPGmodel
import tensorflow as tf
from keras.models import clone_model
from keras.layers import Flatten
class MADDPG:

    def __init__(self, n_agents, state_space, action_space, actor_learning_rate=1e-05, ctor_learning_rate=1e-04, critic_learning_rate=1e-04,
        actor_hidden_units : list = [128],
        critic_hidden_units : list = [128],
        gamma=0.99, tau=0.95, path='') -> None:
        

        self.n_agents =n_agents
        self.state_space = state_space
        self.action_spae = action_space
        self.model : MADDPGmodel = MADDPGmodel(n_agents, state_space, action_space, actor_learning_rate, critic_learning_rate,
            actor_hidden_units, critic_hidden_units)
        self.t_model = clone_model(self.model.model)
        self.gamma = gamma
        self.tau = tau
        self.path = path

       

    def policy(self, states):

        action = self.model.act(states)

        noise = tf.random.uniform(action.shape, -1, 1, seed=7)

        return action + noise

    def learn(self, s, r, a, s_1):
        
        t_q_value = self.t_model(s_1)
        s_f = Flatten()(s)
        a_f = Flatten()(a)
        
        target = tf.math.reduce_mean(r, axis=1) + self.gamma * t_q_value
        self.model.trainCritic(s_f, a_f, target)
        self.model.trainActor(s)

        # Needs Update target
    def updateTarget(self):
        weights = self.model.model.variables
        t_weights = self.t_model.variables

        for a, b in zip(t_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def save(self):
        pass

    def load(self):
        pass
if __name__ == '__main__':
    import numpy as np
    m = MADDPG(2, 2, 2)
    state = np.random.random((1, 2, 2))
    r = np.random.random((1, 2))
    actions = np.random.random((1, 2, 2))
    state_1 = np.random.random((1, 2, 2))
    m.updateTarget()
