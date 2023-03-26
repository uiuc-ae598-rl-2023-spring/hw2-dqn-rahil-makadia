import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import discreteaction_pendulum

# write the DQN algorithm here following Mnih et al. (2015) for the following cases:
# With replay, with target Q (i.e., the standard algorithm).
# With replay, without target Q (i.e., the target network is reset after each step).
# Without replay, with target Q (i.e., the size of the replay memory buffer is equal to the size of each minibatch).
# Without replay, without target Q (i.e., the target network is reset after each step and the size of the replay memory buffer is equal to the size of each minibatch).

class DQN:
    def __init__(self, env, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, batch_size, replay_size, target_update, num_hidden, num_layers):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.target_update = target_update
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.replay_buffer = []
        self.replay_buffer_index = 0

        self.num_actions = env.num_actions
        self.num_states = env.num_states

        self.build_model()

    def build_model(self):
        # build the model here
        # the model should have the following inputs:
        # state
        # action
        # target Q value
        # the model should have the following outputs:
        # Q value
        # the model should have the following losses:
        # mean squared error between the target Q value and the Q value
        # the model should have the following optimizers:
        # RMSProp with learning rate self.learning_rate
        # the model should have the following updates:
        # update the target Q network every self.target_update steps
        # the model should have the following summaries:
        # the loss
        # the epsilon
        # the Q value
        # the target Q value
        # the action
        # the state
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.num_states,), name='input'))
        model.add(tf.keras.layers.Flatten(name='flatten'))
        for i in range(self.num_layers):
            model.add(tf.keras.layers.Dense(self.num_hidden, activation='tanh', name=f'hidden_{i}'))
        model.add(tf.keras.layers.Dense(self.num_actions, activation='linear', name='output'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse')
        self.model = model
        # model.summary()
        self.target_model = model
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.trainable = False
        self.target_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse')
        # self.target_model.summary()
        return None

    def act(self, state):
        # choose an action here
        # if the random number is less than epsilon, choose a random action
        # otherwise, choose the action with the highest Q value
        # return the action
        val = np.random.random()
        if val < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.model.predict(state, verbose=0))

    def observe(self, state, action, reward, next_state, done):
        # add the transition to the replay buffer
        # if the replay buffer is full, replace the oldest transition
        # if the replay buffer is not full, add the transition to the end of the replay buffer
        buffer = (state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.replay_size:
            self.replay_buffer.append(buffer)
        else:
            self.replay_buffer[self.replay_buffer_index] = buffer
            self.replay_buffer_index = (self.replay_buffer_index + 1) % self.replay_size
        return None

    def replay(self):
        # sample a minibatch from the replay buffer
        # train the model on the minibatch
        minibatch_idx = np.random.choice(len(self.replay_buffer), self.batch_size)
        minibatch = [self.replay_buffer[i] for i in minibatch_idx]
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
        self.model.fit(states, target, verbose=0)
        return None

    def check_target_update(self, iter_count):
        # update the target network every self.target_update steps
        if iter_count % self.target_update == 0:
            self.target_model.set_weights(self.model.get_weights())
        return None

    def _run_one_episode(self):
        # run one episode here
        # initialize the state
        # for each step:
        # choose an action
        # take a step
        # observe the transition
        # replay
        # update epsilon
        # if the episode is done, break
        s = self.env.reset()
        done = False
        iter_count = 0
        log = {
            't': [0],
            's': [s],
            'a': [0],
            'r': [0],
            'theta': [self.env.x[0]],
            'thetadot': [self.env.x[1]],
        }
        G = 0
        while not done:
            a = self.act(s)
            s_, r, done = self.env.step(a)
            self.observe(s, a, r, s_, done)
            self.replay()
            G += r*self.gamma**iter_count
            iter_count += 1
            self.check_target_update(iter_count)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            s = s_
            log['t'].append(log['t'][-1] + 1)
            log['s'].append(s)
            log['a'].append(a)
            log['r'].append(r)
            log['theta'].append(self.env.x[0])
            log['thetadot'].append(self.env.x[1])
        log['G'] = G
        tau = [self.env._a_to_u(a) for a in log['a']]
        log['tau'] = tau
        value_func = [np.max(self.model.predict(s, verbose=0)) for s in log['s']]
        log['value_func'] = value_func
        return log

    def run(self, num_episodes):
        # run the DQN algorithm here
        # for each episode:
        # initialize the state
        # for each step:
        # choose an action
        # take a step
        # observe the transition
        # replay
        # update epsilon
        # if the episode is done, break
        self.num_episodes = num_episodes
        self.G = []
        self.all_theta = []
        self.all_thetadot = []
        self.all_tau = []
        self.all_value_func = []
        for i in range(num_episodes):
            print(f'Episode {i+1}')
            log = self._run_one_episode()
            self.G.append(log['G'])
            self.all_theta += log['theta']
            self.all_thetadot += log['thetadot']
            self.all_tau += log['tau']
            self.all_value_func += log['value_func']
        return None

    def plot(self):
        def wrap_pi(x): return ((x + np.pi) % (2 * np.pi)) - np.pi
        def wrap_2pi(x): return x % (2 * np.pi)
        wrap_func = wrap_pi
        log = self._run_one_episode()
        size = 6

        # plot the return and n-episode moving average
        n_avg = 5
        moving_avg = np.convolve(self.G, np.ones((n_avg,))/n_avg, mode='valid')
        plt.figure(figsize=(size, size), dpi=150)
        plt.plot(np.arange(1, self.num_episodes+1), self.G, label='Return', alpha=0.5)
        plt.plot(np.arange(n_avg, self.num_episodes+1), moving_avg, label=f'{n_avg}-episode moving average')
        plt.xlabel('Episode #')
        plt.ylabel('Return')
        plt.legend()
        plt.title('Return vs. Episode #')
        plt.savefig(f'./figures/results/return_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # plot the trajectory
        log['theta'] = [wrap_func(x) for x in log['theta']]
        plt.figure(figsize=(size, size), dpi=150)
        plt.plot(log['t'], log['theta'], label=r'$\theta$')
        plt.axhline(-np.pi, color='r', linestyle='--')
        plt.axhline(np.pi, color='r', linestyle='--', label=r'$\theta=\pm\pi$')
        plt.axhline(-0.1*np.pi, color='g', linestyle='--')
        plt.axhline(0.1*np.pi, color='g', linestyle='--', label=r'$\theta=\pm0.1\pi$')
        plt.plot(log['t'], log['thetadot'], label=r'$\dot{\theta}$')
        plt.xlabel('t')
        plt.ylabel('theta / thetadot')
        plt.legend()
        plt.title('Pendulum State vs. Time')
        plt.savefig(f'./figures/results/state_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')

        # save animation
        policy_lambda = lambda s: np.argmax(self.model.predict(s, verbose=0))
        self.env.video(policy_lambda, f'./figures/results/animation_{self.num_episodes}_{self.target_update}.gif')

        # plot the policy
        plt.figure(figsize=(size, size), dpi=150)
        plt.scatter(self.all_theta, self.all_thetadot, c=self.all_tau, cmap='viridis')
        plt.colorbar()
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title('Policy')
        plt.savefig(f'./figures/results/policy_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # plot the value function
        plt.figure(figsize=(size, size), dpi=150)
        plt.scatter(self.all_theta, self.all_thetadot, c=self.all_value_func, cmap='viridis')
        plt.colorbar()
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-self.env.max_thetadot, self.env.max_thetadot)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot{\theta}$')
        plt.title('Value Function')
        plt.savefig(f'./figures/results/value_func_{self.num_episodes}_{self.target_update}.png', dpi=300, bbox_inches='tight')
        plt.close()
        return None

def main():
    env = discreteaction_pendulum.Pendulum()
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    learning_rate = 0.00025
    batch_size = 32
    replay_size = 10000
    target_update = 5
    num_hidden = 64
    num_layers = 2
    dqn = DQN(env, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, batch_size, replay_size, target_update, num_hidden, num_layers)

    num_episodes = 100
    dqn.run(num_episodes)
    dqn.plot()
    return None

if __name__ == '__main__':
    main()
