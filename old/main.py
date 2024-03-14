from old.dataGetter import DataGetter
from old.env import SingleAssetTradingEnvironment
from replay import ReplayMemory, Transition
import numpy as np
import torch
from old.DQN import DQNAgent, DuellingDQN
from parameters import *
def train():
    # Environment and Agent Initiation

    ## Cryptocurrency Tickers
    # asset_codes = ["ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "DOGE-USD",
    #                "ADA-USD", "MATIC-USD", "AVAX-USD", "WAVES-USD"]
    asset_codes = ["ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD",]
    # asset_codes=["BTC-USD"]
    ## Training and Testing Environments
    assets = [DataGetter(a, start_date="2015-01-01", end_date="2021-05-01") for a in asset_codes]
    test_assets = [DataGetter(a, start_date="2021-05-01", end_date="2022-05-01", freq="1d") for a in asset_codes]
    envs = [SingleAssetTradingEnvironment(a) for a in assets]
    test_envs = [SingleAssetTradingEnvironment(a) for a in test_assets]

    ## Agent

    memory = ReplayMemory()
    agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

    # Main training loop
    N_EPISODES = 20  # No of episodes/epochs
    scores = []
    eps = EPS_START
    act_dict = {0: -1, 1: 1, 2: 0}

    te_score_min = -np.Inf
    for episode in range(1, 1 + N_EPISODES):
        counter = 0
        episode_score = 0
        episode_score2 = 0
        test_score = 0
        test_score2 = 0

        for env in envs:
            score = 0
            state = env.reset()
            state = state.reshape(-1, STATE_SPACE)
            while True:
                actions = agent.act(state, eps)
                action = act_dict[actions]
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape(-1, STATE_SPACE)

                t = Transition(state, actions, reward, next_state, done)
                agent.memory.store(t)
                agent.learn()

                state = next_state
                score += reward
                counter += 1
                if done:
                    break

            episode_score += score
            episode_score2 += (env.store['running_capital'][-1] - env.store['running_capital'][0])

        scores.append(episode_score)
        eps = max(EPS_END, EPS_DECAY * eps)

        for i, test_env in enumerate(test_envs):
            state = test_env.reset()
            done = False
            score_te = 0
            scores_te = [score_te]

            while True:
                actions = agent.act(state)
                action = act_dict[actions]
                next_state, reward, done, _ = test_env.step(action)
                next_state = next_state.reshape(-1, STATE_SPACE)
                state = next_state
                score_te += reward
                scores_te.append(score_te)
                if done:
                    break

            test_score += score_te
            test_score2 += (test_env.store['running_capital'][-1] - test_env.store['running_capital'][0])
        if test_score > te_score_min:
            te_score_min = test_score
            torch.save(agent.actor_online.state_dict(), "online.pt")
            torch.save(agent.actor_target.state_dict(), "target.pt")

        print(f"Episode: {episode}, Train Score: {episode_score:.5f}, Validation Score: {test_score:.5f}")
        print(f"Episode: {episode}, Train Value: ${episode_score2:.5f}, Validation Value: ${test_score2:.5f}", "\n")


if __name__ == '__main__':
    train()

