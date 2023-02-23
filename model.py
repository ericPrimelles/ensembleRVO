from keras import Model, layers
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf


class MADDPGmodel:

    def __init__(self, n_agents, state_space, action_space, actor_learning_rate=1e-05, critic_learning_rate=1e-04,
        actor_hidden_units : list = [128],
        critic_hidden_units : list = [128]
    ) -> None:
        self.n_agents = n_agents

        self.state_space = state_space
        self.action_space = action_space
        self.actor_optimizer = Adam(actor_learning_rate)
        self.critic_optimizer = Adam(critic_learning_rate)
        self.actor_hidden_units = actor_hidden_units
        self.critic_hidden_units = critic_hidden_units
        self.actors_models = [
            self.getDDPGActor()
            for i in range(self.n_agents)
        ]
        self.actor = self.getEnsembleActor()
        self.critic = self.getCritic()
        self.model = self.getEnsembleModel()

    def getDDPGActor(self) -> Sequential:

        return Sequential (
            [
                layers.Input(self.state_space),
                *[layers.Dense(units, activation='relu') for units in self.actor_hidden_units],
                layers.Dense(self.action_space, activation='relu')
            ]
        )
    def getEnsembleActor(self):
        
        inputs = layers.Input((self.n_agents, self.state_space))
        
        actions = tf.concat([
            actor(inputs[:, i, :])
            for i, actor in enumerate(self.actors_models)
        ], axis=1)

        return Model(inputs=inputs, outputs=actions)

    def getCritic(self):

        state = layers.Input(self.n_agents * self.state_space, name='critic_states')
        #state = layers.Flatten()(state)

        actions = layers.Input(self.n_agents * self.action_space, name='critic_actions')
        #actions = layers.Flatten()(actions)

        inputs = layers.Concatenate()([state, actions])
        
        h_v = layers.Dense(self.critic_hidden_units[0], activation='relu')(inputs)

        for indx, units in enumerate(self.critic_hidden_units):
            if indx == 0:
                continue
            
            h_v = layers.Dense(units)(h_v)

        q_value = layers.Dense(1, activation='relu')(h_v)

        return Model(inputs=[state, actions], outputs=q_value) 
        
    def getEnsembleModel(self):
        states = layers.Input((self.n_agents, self.state_space))

        actions = self.actor(states)

        c_states = layers.Flatten()(states)

        q_value = self.critic([c_states,actions])

        return Model(inputs=states, outputs=q_value)

    @tf.function
    def trainCritic(self, states, actions, target):
        self.actor.trainable = False
        with tf.GradientTape() as tape:
            q_value = self.critic([states, actions], training=True)
            loss = tf.math.reduce_mean(tf.math.square(target - q_value))
        
        q_grad = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(q_grad, self.critic.trainable_variables))
        self.actor.trainable=True
        
    @tf.function
    def trainActor(self, states):
        self.critic.trainable = False

        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.model(states))
        a_grad = tape.gradient(loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(a_grad, self.model.trainable_variables))
        self.critic.trainable=True

    @tf.function
    def act (self, state):
        
        return tf.linalg.normalize(self.actor(state), axis=1)[0]

    def save (self):
        pass

    def load(self):
        pass
if __name__ == '__main__':

    import numpy as np
    x = MADDPGmodel(2, 2, 2)
    m = x.model

    states = np.random.random((1, 2, 2))

    target = np.random.random((1, 1))
    
    print(x.act(states))
    