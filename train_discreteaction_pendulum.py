from DQN import DQN

def main():
    env = discreteaction_pendulum.Pendulum()
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    learning_rate = 0.00025
    batch_size = 32
    replay_size = 100000
    init_replay_size = 500
    target_update = 1000
    savefig = True
    dqn = DQN(env, gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig)

    num_episodes = 150
    dqn.run(num_episodes)
    return None

if __name__ == '__main__':
    main()
