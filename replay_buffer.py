import numpy as np
import tensorflow as tf

class ReplayBuffer:
    
    max_length : int = 1000
    size_batch : int = 256
    indx : int = 0
    ready : bool = False
    
    
    def __init__(self, observation_spec, action_spec, n_agents,max_length : int = 1000, batch_size : int = 256) -> None:
        self.max_length = max_length
        self.size_batch = batch_size
        self.n_agents = n_agents
        self.obs_space = observation_spec
        self.act_space = action_spec
        self.ready = False
        self.indx = 0

        #Buffers
        self.s = np.empty((self.max_length, 1, self.n_agents, self.obs_space))
        self.a = np.empty((self.max_length, 1, self.n_agents, self.act_space))
        self.s_1 = np.empty((self.max_length, 1, self.n_agents, self.obs_space))  
        self.r = np.empty((self.max_length, 1, self.n_agents))
        self.dones = np.empty((self.max_length, 1, self.n_agents))
        
    def reshape(self, s, a, r, s_1, done):
        s = np.reshape(s, (1, self.n_agents, self.obs_space))
        a = np.reshape(a, (1, self.n_agents, self.act_space))
        s_1 = np.reshape(s_1, (1, self.n_agents, self.obs_space))
        r = np.reshape(r, (1, self.n_agents))
        dones = np.reshape(done, (1, self.n_agents))
        return s, a, r, s_1, dones

    def store(self, s, a, r, s_1, done) -> None:
        s, a, r, s_1, done = self.reshape(s, a, r, s_1, done)
        
        i = self.indx % self.max_length
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s_1[i] = s_1
        self.dones[i] = done
        self.indx += 1

        if self.indx == self.size_batch:
            self.ready = True
    
    def sample(self) -> list:
        if not self.ready:
            return []
        i = 0
        if self.indx < self.max_length: 
            i = self.indx
        else:
            i = self.max_length    
        
        
        sample = np.random.randint(0, i, self.size_batch)
        s = [self.s[i] for i in sample]
        a = [self.a[i] for i in sample]
        r = [self.r[i] for i in sample]
        s_1 = [self.s_1[i] for i in sample]
        done = [self.dones[i] for i in sample]
        
        s = np.concatenate(s, 0)
        a = np.concatenate(a, 0)
        r = np.concatenate(r, 0)
        s_1 = np.concatenate(s_1, 0)
        done = np.concatenate(done, 0)
        return tf.convert_to_tensor(s), tf.convert_to_tensor(a), tf.convert_to_tensor(r), tf.convert_to_tensor(s_1), tf.convert_to_tensor(done)
    
    def isReady(self) -> bool:

        return self.ready

if __name__ =='__main__':
    from pettingzoo.mpe import simple_v2

    env = simple_v2.env(continuous_actions=True)         
    rb = ReplayBuffer(4, 5, 1, 100, 20)
    env.reset()
    s = env.observe('agent_0')
   
    while not rb.isReady():
        env.render()
        a = np.random.random(5)
        env.step(a)
        
        s_1, r, done, truncation, info = env.last()
        rb.store(s, a, r, s_1, done)
        s = s_1
        if done or truncation:
            env.reset()
    s, a, r, s_1, d = rb.sample()
    print(s.shape)