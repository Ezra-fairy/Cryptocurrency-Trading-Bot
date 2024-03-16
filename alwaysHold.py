from dataGetter import DataGetter
from env import Environment
from replay import *
from parameters import *
from DQN import DQNAgent, DuellingDQN
from lineGraph import drawLineGraph
import numpy as np

def train():
    # Setting up environment and dataset
    asset_name = "ETH-USD"
    asset = DataGetter(asset=asset_name, start_date="2017-11-09", end_date="2023-03-09")
    test_asset = DataGetter(asset=asset_name, start_date="2023-03-09", end_date="2024-03-09")
    env = Environment(asset)
    test_env = Environment(test_asset)

    # Initiate Agent
    memory = ReplayMemory()
    agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

    # Main training loop
    N_EPISODES = 10
    all_scores = []
    all_capitals = []
    all_test_scores = []
    all_test_capitals = []
    eps = EPS_START
    act_dict = {0: -1, 1: 1, 2: 0}
    act_explain_dict = {0: "Sell", 1: "Buy", 2: "Hold"}

    # Store actions into a numpy array.
    trainActionArray = []
    testActionArray = []
    train_terminate_index = 0

    # te_score_min = -np.Inf
    for episode in range(1, 1 + N_EPISODES):
        episode_rewards = []
        capital_left = 0

        state = env.reset()
        state = state.reshape(-1, STATE_SPACE)
        while True:
            actions = 1
            action = act_dict[actions]

            # Only store the action to draw graph in the last episode
            if episode == N_EPISODES:
                trainActionArray.append(action)

            next_state, reward, terminal, total_capital, index = env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)

            # Convert terminal into Boolean value dones then store it into Transition because we need boolean while updating networks
            done = terminal != 0

            state = next_state
            episode_rewards.append(reward)
            if terminal != 0:
                print("terminate with code ", terminal)
                capital_left = total_capital - 100000
                train_terminate_index = index
                break


        ave_reward  = sum(episode_rewards) / len(episode_rewards)

        all_scores.append(ave_reward)
        all_capitals.append(capital_left)


        # ------------------------------------------------TEST---------------------------------------------------------------------
        state = test_env.reset()
        test_episode_rewards = []
        test_capital_left = 0

        while True:
            actions = 1
            # print("Chose Action", act_explain_dict[actions])
            action = act_dict[actions]

            # Only store the action to draw graph in the last episode
            if episode == N_EPISODES:
                testActionArray.append(action)

            next_state, test_reward, terminal, test_total_capital, index = test_env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)
            state = next_state
            test_episode_rewards.append(test_reward)

            if terminal != 0:
                test_capital_left = test_total_capital - 100000
                test_terminate_index = index
                break

        ave_test_reward = sum(test_episode_rewards) / len(test_episode_rewards)

        all_test_scores.append(ave_test_reward)
        all_test_capitals.append(test_capital_left)

        print(f"Episode: {episode}, Epsilon: {eps:.2f} Average Train Reward: {ave_reward:.5f}, On Test Set: {ave_test_reward:.5f}")
        print(f"Episode: {episode}, Epsilon: {eps:.2f} Capital Left Training: ${capital_left:.5f}, On Test Set: ${test_capital_left:.5f}", "\n")


        eps = max(EPS_END, EPS_DECAY * eps)
        # if test_reward > te_score_min:
        #     te_score_min = test_reward
        #     torch.save(agent.actor_online.state_dict(), "online.pt")
        #     torch.save(agent.actor_target.state_dict(), "target.pt")
    print(f"Average Training Capital left: {sum(all_capitals) / len(all_capitals)}")
    print(f"Average Testing Capital left: {sum(all_test_capitals) / len(all_test_capitals)}")
    print(f"Average Training Reward Got: {sum(all_scores) / len(all_scores)}")
    print(f"Average Testing Reward Got: {sum(all_test_scores) / len(all_test_scores)}")
    # Draw the graph
    drawLineGraph(asset.dateArray[:train_terminate_index+1], asset.priceArray[:train_terminate_index+1], trainActionArray, "Train.png")
    drawLineGraph(test_asset.dateArray, test_asset.priceArray, testActionArray, "Test.png")

if __name__ == '__main__':
    train()