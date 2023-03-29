from DQN import DQN
import time

def main():
    num_episodes = 150
    verbose = False
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    learning_rate = 0.00025
    batch_size = 32
    replay_size = 100000
    init_replay_size = 500
    target_update = 1000
    savefig = True

    start = time.time()
    target_Q = True
    replay = True    
    dqn_yes_target_yes_replay = DQN(gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig, target_Q, replay, verbose)
    dqn_yes_target_yes_replay.run(num_episodes)
    end = time.time()
    print(f"Time taken for DQN with target network and replay buffer: {(end - start)/60:0.3f} minutes")

    start = time.time()
    target_Q = True
    replay = False
    dqn_yes_target_no_replay = DQN(gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig, target_Q, replay, verbose)
    dqn_yes_target_no_replay.run(num_episodes)
    end = time.time()
    print(f"Time taken for DQN with target network and no replay buffer: {(end - start)/60:0.3f} minutes")

    start = time.time()
    target_Q = False
    replay = True
    dqn_no_target_yes_replay = DQN(gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig, target_Q, replay, verbose)
    dqn_no_target_yes_replay.run(num_episodes)
    end = time.time()
    print(f"Time taken for DQN with no target network and replay buffer: {(end - start)/60:0.3f} minutes")

    start = time.time()
    target_Q = False
    replay = False
    dqn_no_target_no_replay = DQN(gamma, epsilon, epsilon_min, learning_rate, batch_size, replay_size, init_replay_size, target_update, savefig, target_Q, replay, verbose)
    dqn_no_target_no_replay.run(num_episodes)
    end = time.time()
    print(f"Time taken for DQN with no target network and no replay buffer: {(end - start)/60:0.3f} minutes")
    return None

if __name__ == '__main__':
    main()
