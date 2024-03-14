from dataGetter import DataGetter
from env import Environment
from replay import *
from parameters import *
import numpy as np

def train():
    # Setting up environment and dataset
    asset_name = "ETH-USD"
    asset = DataGetter(asset=asset_name, start_date="2017-11-09", end_date="2023-03-06")
    test_asset = DataGetter(asset=asset_name, start_date="2023-03-06", end_date="2024-03-06")
    env = Environment(asset)
    test_env = Environment(test_asset)


    # Main training loop
    N_EPISODES = 200
    all_scores = []
    all_capitals = []
    all_test_scores = []
    all_test_capitals = []
    eps = EPS_START
    act_dict = {0: -1, 1: 1, 2: 0}

    # te_score_min = -np.Inf
    for episode in range(1, 1 + N_EPISODES):
        episode_rewards = []
        capital_left = 0

        state = env.reset()
        state = state.reshape(-1, STATE_SPACE)
        while True:
            actions = 1
            action = act_dict[actions]
            next_state, reward, terminal, total_capital = env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)

            # Convert terminal into Boolean value dones then store it into Transition because we need boolean while updating networks
            done = terminal != 0

            t = Transition(state, actions, reward, next_state, done)

            state = next_state
            episode_rewards.append(reward)
            if terminal != 0:
                print("terminate with code ", terminal)
                capital_left = total_capital - 100000
                break


        ave_reward  = sum(episode_rewards) / len(episode_rewards)

        all_scores.append(ave_reward)
        all_capitals.append(capital_left)
        eps = max(EPS_END, EPS_DECAY * eps)


        state = test_env.reset()
        test_episode_rewards = []
        test_capital_left = 0

        while True:
            actions = 1
            action = act_dict[actions]
            next_state, test_reward, terminal, test_total_capital = test_env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)
            state = next_state
            test_episode_rewards.append(test_reward)

            if terminal != 0:
                test_capital_left = test_total_capital - 100000
                break

        ave_test_reward = sum(test_episode_rewards) / len(test_episode_rewards)

        all_test_scores.append(ave_test_reward)
        all_test_capitals.append(test_capital_left)

        print(f"Episode: {episode}, Epsilon: {eps:.2f} Average Train Reward: {ave_reward:.5f}, On Test Set: {ave_test_reward:.5f}")
        print(f"Episode: {episode}, Epsilon: {eps:.2f} Capital Left Training: ${capital_left:.5f}, On Test Set: ${test_capital_left:.5f}", "\n")

    print(f"Average Training Capital left: {sum(all_capitals)/len(all_capitals)}")
    print(f"Average Testing Capital left: {sum(all_test_capitals) / len(all_test_capitals)}")
    print(f"Average Training Reward Got: {sum(all_scores)/len(all_scores)}")
    print(f"Average Testing Reward Got: {sum(all_test_scores) / len(all_test_scores)}")

if __name__ == '__main__':
    train()