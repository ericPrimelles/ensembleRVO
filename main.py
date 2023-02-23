from MADDPG import MADDPG
from replay_buffer import ReplayBuffer
from pettingzoo.mpe import simple_v2
import tensorflow as tf
gamma = 0.99
tau = 0.95
actor_hidden_units = [128, 64, 32]
critic_hidden_units = [64, 32, 16]



if __name__ == '__main__':
    env = simple_v2.env(max_cycles=50, continuous_actions=True)
    m = MADDPG(1, 4, 5, gamma=gamma, tau=tau, actor_hidden_units=actor_hidden_units, critic_hidden_units=critic_hidden_units)
    rb = ReplayBuffer(4, 5, 1)

    for i in range(1000):
        env.reset()
        s = env.observe('agent_0')

        while 1:
            s_e = tf.expand_dims(s, 0)
            s_e = tf.expand_dims(s_e, 0)
            
            a = m.policy(s_e)                    
            #a = self.env.sample()
            a = tf.squeeze(a)
            env.step(a)                    
            s_1, r, done, truncation, info = env.last()
            rb.store(s, a, r, s_1, done)
            if rb.ready:
                s, a, r, s_1, done = rb.sample()
                m.learn(s, a, r, s_1)

            if i % 1000 == 0:
                m.updateTarget()                                                
            s = s_1
            

            if truncation or done:
                #env.reset()
                print(f'Epoch {i} ended')
                break  
                
