import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import discreteaction_pendulum

import torch
import torch.nn as nn
import torch.optim as optim

# write the DQN algorithm here following Mnih et al. (2015) for the following cases:
# With replay, with target Q (i.e., the standard algorithm).
# With replay, without target Q (i.e., the target network is reset after each step).
# Without replay, with target Q (i.e., the size of the replay memory buffer is equal to the size of each minibatch).
# Without replay, without target Q (i.e., the target network is reset after each step and the size of the replay memory buffer is equal to the size of each minibatch).

class DQN:
    def __init__(self, env, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, num_hidden, num_layers):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.target_update = target_update
        self.target_counter = 0
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.replay_buffer = []
        self.init_replay_size = init_replay_size
        self.initialize_replay_buffer()

        self.num_actions = env.num_actions
        self.num_states = env.num_states

        self.build_model()
        return None

    def initialize_replay_buffer(self):
        while len(self.replay_buffer) < self.init_replay_size:
            s = self.env.reset()
            done = False
            while not done:
                a = np.random.randint(self.env.num_actions)
                (s_, r, done) = self.env.step(a)
                self.replay_buffer.append((s, a, r, s_, done))
                s = s_
        return None

    # def build_model(self):
    #     # build the model here
    #     # update the target Q network every self.target_update steps
    #     model = tf.keras.models.Sequential()
    #     model.add(tf.keras.layers.InputLayer(input_shape=(self.num_states,), name='input'))
    #     model.add(tf.keras.layers.Flatten(name='flatten'))
    #     for i in range(self.num_layers - 1):
    #         model.add(tf.keras.layers.Dense(self.num_hidden, activation='tanh', name=f'hidden_{i}'))
    #     model.add(tf.keras.layers.Dense(self.num_actions, activation='linear', name='output'))
    #     model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse') ## , rho=0.95, epsilon=0.01, momentum=0.95
    #     self.model = model
    #     # model.summary()
    #     self.target_model = model
    #     self.target_model.set_weights(self.model.get_weights())
    #     # self.target_model.trainable = False
    #     self.target_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse') # , rho=0.95, epsilon=0.01, momentum=0.95
    #     # self.target_model.summary()
    #     return None

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.env.num_states, self.num_hidden),
            nn.Tanh(),
            # nn.Linear(self.num_hidden, self.num_hidden),
            # nn.Tanh(),
            nn.Linear(self.num_hidden, self.env.num_actions)
        )
        self.model = model
        self.target_model = model
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=0.95, eps=0.01, momentum=0.95)
        self.loss_fn = nn.MSELoss()
        return None

    def act(self, state):
        val = np.random.random()
        if val < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            # return np.argmax(self.model.predict(state, verbose=0)) # for tf.keras
            return np.argmax(self.model(torch.from_numpy(state).float()).detach().numpy()) # for pytorch

    def observe(self, state, action, reward, next_state, done):
        # if the replay buffer is full, replace the oldest transition
        # if the replay buffer is not full, add the transition to the end of the replay buffer
        buffer = (state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.replay_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(buffer)
        return None

    # def replay(self):
    #     # sample a minibatch from the replay buffer
    #     # train the model on the minibatch
    #     minibatch_idx = np.random.choice(len(self.replay_buffer), self.batch_size)
    #     minibatch = [self.replay_buffer[i] for i in minibatch_idx]
    #     states = np.array([x[0] for x in minibatch])
    #     actions = np.array([x[1] for x in minibatch])
    #     rewards = np.array([x[2] for x in minibatch])
    #     next_states = np.array([x[3] for x in minibatch])
    #     dones = np.array([x[4] for x in minibatch])
    #     target = self.model.predict(states, verbose=0)
    #     # target_model_temp = tf.keras.models.clone_model(self.model)
    #     # target_next = target_model_temp.predict(next_states, verbose=0)
    #     target_next = self.target_model.predict(next_states, verbose=0)
    #     for i in range(self.batch_size):
    #         if dones[i]:
    #             target[i][actions[i]] = rewards[i]
    #         else:
    #             target[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
    #     self.model.fit(states, target, verbose=0, epochs=1)
    #     return None

    def replay(self):
        batch_idx = np.random.choice(len(self.replay_buffer), self.batch_size)
        batch = [self.replay_buffer[i] for i in batch_idx]
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        q_values = self.model(torch.FloatTensor(states))
        next_q_values = self.target_model(torch.FloatTensor(next_states)).detach()
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])
        loss = self.loss_fn(q_values, q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return None

    # def check_target_update(self):
    #     # update the target network every self.target_update steps
    #     if self.target_counter % self.target_update == 0:
    #         self.target_model.set_weights(self.model.get_weights())
    #     return None

    def check_target_update(self):
        if self.target_counter % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return None

    def _run_one_episode(self):
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
            self.target_counter += 1
            G += r*self.gamma**iter_count
            iter_count += 1
            self.check_target_update()
            self.epsilon = max(self.epsilon - 0.9/1e4, self.epsilon_min)
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
        # value_func = [np.max(self.model.predict(s, verbose=0)) for s in log['s']] # for keras
        value_func = [torch.max(self.model(torch.FloatTensor(s))).item() for s in log['s']] # for pytorch
        log['value_func'] = value_func
        self.G.append(log['G'])
        self.all_theta += log['theta']
        self.all_thetadot += log['thetadot']
        self.all_tau += log['tau']
        self.all_value_func += log['value_func']
        self.plot(log)
        return log

    def run(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_list = [1]
        self.G = []
        self.all_theta = []
        self.all_thetadot = []
        self.all_tau = []
        self.all_value_func = []
        for i in range(num_episodes):
            log = self._run_one_episode()
            print(f'Episode {i+1}, Return: {log["G"]}, Epsilon: {self.epsilon}')
            self.episode_list.append(i+2)
        return None

    def plot(self, log=None):
        def wrap_pi(x): return ((x + np.pi) % (2 * np.pi)) - np.pi
        def wrap_2pi(x): return x % (2 * np.pi)
        wrap_func = wrap_pi
        # save animation
        if log is None:
            log = self._run_one_episode()
            # policy_lambda = lambda s: np.argmax(self.model.predict(s, verbose=0)) # for keras
            policy_lambda = lambda s: np.argmax(self.model(torch.FloatTensor(s)).detach().numpy()) # for pytorch
            self.env.video(policy_lambda, f'./figures/results/animation_{self.num_episodes}_{self.target_update}.gif')
        size = 6

        # plot the return and n-episode moving average
        n_avg = 10
        moving_avg = np.convolve(self.G, np.ones((n_avg,))/n_avg, mode='valid')
        plt.figure(figsize=(size, size), dpi=150)
        plt.plot(self.episode_list, self.G, label='Return', alpha=0.3)
        if max(self.episode_list) > n_avg:
            plt.plot(self.episode_list[n_avg-1:], moving_avg, label=f'{n_avg}-episode moving average')
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
        plt.close()

        # size = 20
        # theta_vec = np.linspace(-np.pi, np.pi, size)
        # thetadot_vec = np.linspace(-self.env.max_thetadot, self.env.max_thetadot, size)
        # theta_grid, thetadot_grid = np.meshgrid(theta_vec, thetadot_vec)
        # policy_arr = np.zeros((size, size))
        # value_func_arr = np.zeros((size, size))
        # for i in range(size):
        #     for j in range(size):
        #         s = np.array([theta_grid[i, j], thetadot_grid[i, j]])
        #         predict = self.model.predict(s, verbose=0)
        #         policy_arr[i, j] = np.argmax(predict)
        #         value_func_arr[i, j] = np.max(predict)
        # plot the policy
        plt.figure(figsize=(size, size), dpi=150)
        plt.scatter(self.all_theta, self.all_thetadot, c=self.all_tau, cmap='viridis')
        # plt.imshow(policy_arr, cmap='viridis', extent=[-np.pi, np.pi, -self.env.max_thetadot, self.env.max_thetadot])
        # plt.contourf(theta_grid, thetadot_grid, policy_arr, cmap='viridis')
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
        # plt.imshow(value_func_arr, cmap='viridis', extent=[-np.pi, np.pi, -self.env.max_thetadot, self.env.max_thetadot])
        # plt.contourf(theta_grid, thetadot_grid, value_func_arr, cmap='viridis')
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
    replay_size = 100000
    init_replay_size = 5000
    target_update = 1000
    num_hidden = 64
    num_layers = 2
    dqn = DQN(env, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, num_hidden, num_layers)

    num_episodes = 100
    dqn.run(num_episodes)
    dqn.plot()
    return None

if __name__ == '__main__':
    main()
