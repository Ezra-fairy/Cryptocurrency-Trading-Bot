import numpy as np
class Environment:
    def __init__(self, data, initial = 100000, trade_amount = 5000,
                 trans_cost=0.005, cap_low=0.1, cap_high = 10):
        self.data = data

        self.initial = initial
        self.total_capital = initial
        self.cash = initial
        self.asset_holding = 0
        self.trade_amount = trade_amount

        self.trans_cost = trans_cost
        self.cap_low = cap_low
        self.cap_high = cap_high

        self.terminal_idx = len(self.data) - 1
        self.cur_idx = 0
        self.terminal = 0 # 0 is not terminal, 1 is up, 2, is down, 3 is end

        self.state = None
        self.action = 0
        self.reward = 0
        self.current_price = self.data.frame.iloc[self.cur_idx, :]['Adj Close']

    def reset(self):
        self.total_capital = self.initial
        self.cash = self.initial
        self.asset_holding = 0

        self.cur_idx = 0
        self.terminal = 0

        self.state = self.get_state(self.cur_idx)
        self.action = 0
        self.reward = 0
        self.current_price = self.data.frame.iloc[self.cur_idx, :]['Adj Close']
        return self.state

    def step(self, action):
        self.action = action
        self.reward = self.calculate_reward()
        self.terminal = self.check_terminal()

        if self.terminal != 0:
            # If we have enough profit and quit
            if self.terminal== 1:
                self.reward = 10
                # also give additional reward for the speed
                self.reward += 5 * (1 - self.cur_idx / self.terminal_idx)
            # If we lose most money and quit
            elif self.terminal==2:
                self.reward = -10
                # Also penalize if lose it too soon
                self.reward += -5 * (1 - self.cur_idx / self.terminal_idx)
            # If end of episode
            if self.cur_idx < self.terminal_idx:
                # penalize it for going too long but still depend on the total cap
                # if 1.5 times initial, we can take it as average
                self.reward = self.total_capital / self.initial - 1.5

        if self.terminal==0:
            self.cur_idx += 1
            self.current_price = self.data.frame.iloc[self.cur_idx, :]['Adj Close']
            self.state = self.get_state(self.cur_idx)
        return self.state, self.reward, self.terminal, self.total_capital

    def calculate_reward(self):
        # Buy Action
        if self.action == 1:
            if self.cash >= self.trade_amount:
                self.asset_holding += self.trade_amount*(1-self.trans_cost) / self.current_price
                self.cash -= self.trade_amount
            elif self.cash > 0:
                self.asset_holding += self.cash*(1-self.trans_cost) / self.current_price
                self.cash = 0

        # Sell Action
        if self.action == -1:
            # if money larger than our unit of trade
            if self.asset_holding*self.current_price >= self.trade_amount:
                self.cash += self.trade_amount*(1-self.trans_cost)
                self.asset_holding -= self.trade_amount / self.current_price
            # if not enough, sell all
            elif self.asset_holding > 0:
                self.cash += self.asset_holding * self.current_price * (1 - self.trans_cost)
                self.asset_holding = 0


        new_total_value = self.cash + self.asset_holding * self.current_price
        reward = 100 * (new_total_value - self.total_capital)/self.total_capital
        self.total_capital = new_total_value

        # penalize holding
        if self.action == 0:
            reward -= 0.1

        return reward
    def get_state(self, idx):
        state = self.data[idx][:]
        # print("Adj Close", self.data[idx][0])
        state = self.data.scaler.transform(state.reshape(1, -1))
        state = np.concatenate([state, [[self.total_capital / self.initial,
                                         self.cash / self.total_capital,
                                         self.asset_holding * self.current_price / self.total_capital,
                                         ]]], axis=-1)
        return state

    def check_terminal(self):
        if self.cur_idx >= self.terminal_idx:  # End of data
            return 3

        lower_bound = self.initial * self.cap_low
        upper_bound = self.initial * self.cap_high
        if self.total_capital <= lower_bound:  # Stop loss
            return 2
        elif self.total_capital >= upper_bound:  # Take profit
            return 1
        else:
            return 0
