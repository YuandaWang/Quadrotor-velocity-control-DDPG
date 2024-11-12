

import numpy as np
import tensorflow as tf
import random

class DDPG_Agent():
    def __init__(self, settings, session):
        # parameters 
        self.buffer_size = settings['DDPG']['replay_buffer_size']
        self.batch_size = settings['DDPG']['batch_size']
        self.gamma = settings['DDPG']['gamma']
        self.dim_s = settings['num_state']
        self.dim_a = settings['num_action']
        self.rl_a = settings['DDPG']['learning_rate_A']
        self.rl_c = settings['DDPG']['learning_rate_C']
        self.tau = settings['DDPG']['soft_update_rate']
        
        self.fc1num = settings['DDPG']['num_fc1']
        self.fc2num = settings['DDPG']['num_fc2']
        
        # TensorFlow Network
        self.sess = session
        # input place holders
        self.S = tf.placeholder(tf.float32, [None, self.dim_s], 'State')
        self.S1 = tf.placeholder(tf.float32, [None, self.dim_s], 'newState')
        self.R = tf.placeholder(tf.float32, [None, 1], 'reward')
        
        # Networks
        self.A   = self.new_actor_net(self.S, 'actorMain')
        self.At  = self.new_actor_net(self.S1, 'actorTarget')
        self.Q   = self.new_critic_net(self.S, self.A, 'criticMain')
        self.Qt  = self.new_critic_net(self.S1, self.At, 'criticTarget')
        
        # param group
        # should be tf.GraphKeys.GLOBAL_VARIABLES in future version of TF
        self.A_param  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actorMain')
        self.At_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actorTarget')
        self.Q_param  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='criticMain')
        self.Qt_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='criticTarget')
        
        # loss functions of Critic
        target_Q = self.R + self.gamma * self.Qt
        td_error = self.Q - target_Q

        # no L2 regularization this time
        self.loss_c = tf.reduce_mean(tf.square(td_error))
        
        # average Q value
        self.avgQ = tf.reduce_mean(self.Q)
        
        self.trainer_critic = tf.train.AdamOptimizer(self.rl_c).minimize(self.loss_c, var_list=self.Q_param)
        
        # loss function of Actor
        loss_a =  tf.reduce_mean(-self.Q)
        self.trainer_actor = tf.train.AdamOptimizer(self.rl_a).minimize(loss_a, var_list=self.A_param)
        
        # soft update to target networks operations
        # tf.multiply in future version
        
        ### ONLY ONE ACTOR NETWORK
        # for Actor
        self.update_actorTarget = \
        [self.At_param[i].assign(tf.multiply(self.A_param[i], 1) + tf.multiply(self.At_param[i], 0))
         for i in range(len(self.At_param))]
        # for Critic
        self.update_criticTarget = \
        [self.Qt_param[i].assign(tf.multiply(self.Q_param[i], self.tau) + tf.multiply(self.Qt_param[i], (1-self.tau)))
         for i in range(len(self.Qt_param))]
        
        self.Q_distance = \
        tf.reduce_sum([tf.reduce_sum(tf.square(self.Qt_param[i] - self.Q_param[i])) for i in range(len(self.Qt_param))])
        
        self.A_distance = \
        tf.reduce_sum([tf.reduce_sum(tf.square(self.At_param[i] - self.A_param[i])) for i in range(len(self.At_param))])
        

        self.clone_actorTarget  = [self.At_param[i].assign(self.A_param[i]) for i in range(len(self.At_param))]
        self.clone_criticTarget = [self.Qt_param[i].assign(self.Q_param[i]) for i in range(len(self.Qt_param))]
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # sync target network and main network
        self.sess.run(self.clone_actorTarget)
        self.sess.run(self.clone_criticTarget)
        
        # replay buffer
        self.exp_buffer = []
        self.buffer_ready = False

    
    # functions:
    # training critic and actor and soft update
    def train_it(self):
        s, a, r, s1, d = self.buffer_sample()
        # train critic network
        # here the A should be assign with stored batch a
        _, loss=self.sess.run([self.trainer_critic, self.loss_c], feed_dict={self.S:s, self.A:a, self.R:r, self.S1:s1})
        # train actor network
        self.sess.run(self.trainer_actor, feed_dict={self.S:s})
        # soft update target network from main network
        self.sess.run(self.update_actorTarget)
        self.sess.run(self.update_criticTarget)
        
        # avgQ = self.sess.run(self.avgQ, feed_dict={self.S:s, self.A:a})
        rawQ = self.sess.run(self.Q, feed_dict={self.S:s, self.A:a})
        
        return loss, rawQ

    # critic networks and actor networks
    def new_critic_net(self, s, a, scope):
        with tf.variable_scope(scope):
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            
            W1_a = tf.Variable(tf.random_normal([self.dim_a, self.fc1num] ,stddev=0.01))
            b1_a = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            
            W2 = tf.Variable(tf.random_normal([self.fc1num, 1] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([1] ,stddev=0.01))
            
            # !! s and a should be linked together in the critic network, or it will not work
            fc1 = tf.sigmoid(tf.matmul(s, W1) + b1 + tf.matmul(a, W1_a) + b1_a)
            out = tf.matmul(fc1, W2) + b2
            
            return out
            
            
    def new_actor_net(self, s, scope):
        with tf.variable_scope(scope):
            W1 = tf.Variable(tf.random_normal([self.dim_s, self.fc1num] ,stddev=0.01))
            b1 = tf.Variable(tf.random_normal([self.fc1num] ,stddev=0.01))
            
            W2 = tf.Variable(tf.random_normal([self.fc1num, self.dim_a] ,stddev=0.01))
            b2 = tf.Variable(tf.random_normal([self.dim_a] ,stddev=0.01))
            
            fc1 = tf.sigmoid(tf.matmul(s, W1) + b1)
            out = tf.sigmoid(tf.matmul(fc1, W2) + b2)
            return out
    

    def buffer_add(self, s, a, r, s1, d):
        experience = np.reshape(np.array([s, a, r, s1, d]),[1,5])
        if len(self.exp_buffer) + len(experience) >= self.buffer_size:
            self.exp_buffer[0:(len(experience) + len(self.exp_buffer))-self.buffer_size] = []
        if len(self.exp_buffer) > 100 * self.batch_size:
            self.buffer_ready = True
            
        self.exp_buffer.extend(experience)
        
    
               
    def buffer_sample(self):
        batch_size = self.batch_size
        if self.batch_size > len(self.exp_buffer):
            print 'Not enough experience, now we only have', len(self.exp_buffer)
            batch_size = 1
        batch = np.reshape(np.array(random.sample(self.exp_buffer, batch_size)), [batch_size, 5])
        s_batch  = np.vstack(batch[:, 0])
        a_batch  = np.vstack(batch[:, 1])
        r_batch  = np.vstack(batch[:, 2])
        s1_batch = np.vstack(batch[:, 3])
        d_batch  = np.vstack(batch[:, 4])
        
        return s_batch, a_batch, r_batch, s1_batch, d_batch
    
    
        
    def buffer_dump(self):
        self.exp_buffer = []
        
