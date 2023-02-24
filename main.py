from MADDPG import MADDPG
from replay_buffer import ReplayBuffer
from pettingzoo.mpe import simple_v2
import tensorflow as tf
import warnings
from time import sleep
warnings.filterwarnings('ignore')
gamma = 0.99
tau = 0.95
actor_hidden_units = [128, 64, 32]
critic_hidden_units = [64, 32, 16]



if __name__ == '__main__':
    env = simple_v2.env(max_cycles=100, continuous_actions=True, render_mode='human')
    m = MADDPG(1, 4, 5, gamma=gamma, tau=tau, actor_hidden_units=actor_hidden_units, critic_hidden_units=critic_hidden_units, path='models/')
    rb = ReplayBuffer(4, 5, 1)

    '''for i in range(1000):
        env.reset()
        s = env.observe('agent_0')

        while 1:
            s_e = tf.expand_dims(s, 0)
            s_e = tf.expand_dims(s_e, 0)
            
            a = m.policy(s_e)
                                
            #a = self.env.sample()
            a = tf.squeeze(a)
            env.step(a)                    
            s_1, r, d, truncation, info = env.last()
            rb.store(s, a, r, s_1, d)
            if rb.ready:
                state, actions, rewards, states_1, done = rb.sample()
                m.learn(state, actions, rewards, states_1)

            if i % 100 == 0:
                m.updateTarget()
                m.save()                                                
            s = s_1
            

            if truncation or d:
                #env.reset()
                print(f'Epoch {i} ended')
                break  '''
    m.load()
    env.reset()
    obs, r, truncation, done, info = env.last()
    while not done and not truncation:
        env.render()
        sleep(0.3)
        s_e = tf.expand_dims(obs, 0)
        s_e = tf.expand_dims(s_e, 0)        
        a = m.model.act(s_e)
        print(tf.squeeze(a))
        env.step(tf.squeeze(a))
        obs, r, truncation, done, info = env.last()
                
    env.close()