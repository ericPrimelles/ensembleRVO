from model import MADDPGmodel
import tensorflow as tf
from keras.models import clone_model
from keras.layers import Flatten
from noise import OUActionNoise
import os
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
        self.path = os.path.join(path)
        
        self.noise = OUActionNoise(np.zeros(self.n_agents), np.ones(self.n_agents) * 0.2)
       

    def policy(self, states):

        action = self.model.act(states)

       

        
        noise = self.noise()
        return action + noise

        
    def learn(self, s, a, r, s_1):
        
        t_q_value = self.t_model(s_1)
        s_f = Flatten()(s)
        a_f = Flatten()(a)
        
        
        rwd = tf.cast(tf.math.reduce_mean(r, axis=1, keepdims=True), tf.float32)
        t = self.gamma * t_q_value
        target =  rwd + t
        self.model.trainCritic(s_f, a_f, target)
        self.model.trainActor(s)

        # Needs Update target
    def updateTarget(self):
        weights = self.model.model.variables
        t_weights = self.t_model.variables

        for a, b in zip(t_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def save(self):
        self.model.save(self.path)

    def load(self):
        self.model.load(self.path)
if __name__ == '__main__':
    import numpy as np
    m = MADDPG(2, 2, 2)
    state = np.random.random((1, 2, 2))
    r = np.random.random((1, 2))
    actions = np.random.random((1, 2, 2))
    state_1 = np.random.random((1, 2, 2))
    print(m.model.act(state))
    print(m.model.actors_models[0](state[0,0,:]))