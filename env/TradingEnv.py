import random
import gym
from gym import spaces
import numpy as np

from render.StockTradingGraph import StockTradingGraph

MAX_NUM_PURCHASE = 5

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_WINDOW_SIZE = 10
LOOKFORWARD_WINDOW_SIZE = 10


def factor_pairs(val):
    return [(i, val / i) for i in range(1, int(val ** 0.5) + 1) if val % i == 0]


class TradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['live', 'file', 'none']}
    visualization = None

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = self._adjust_prices(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + LOOKFORWARD_WINDOW_SIZE + 2), dtype=np.float16)

    def _adjust_prices(self, df):
        adjust_ratio = df['Adjusted_Close'] / df['Close']

        df['Open'] = df.loc[:, 'Open'] * adjust_ratio
        df['High'] = df.loc[:, 'High'] * adjust_ratio
        df['Low'] = df.loc[:, 'Low'] * adjust_ratio
        df['Close'] = df.loc[:, 'Close'] * adjust_ratio

        return df

    def _next_observation(self):

        # in case we are at the very beginning of the timeline
        start = self.current_step - LOOKBACK_WINDOW_SIZE
        start = 0 if start < 0 else start

        end = self.current_step + 1 + LOOKFORWARD_WINDOW_SIZE
        end = self.df.shape[0] if end > self.df.shape[0] else end

        # Get the stock data points for the last 5 days and scale to between 0-1
        state = np.array([
            self.df.loc[range(start, end), 'Open']
                .values / MAX_SHARE_PRICE,
            self.df.loc[range(start, end), 'High']
                .values / MAX_SHARE_PRICE,
            self.df.loc[range(start, end), 'Low']
                .values / MAX_SHARE_PRICE,
            self.df.loc[range(start, end), 'Close']
                .values / MAX_SHARE_PRICE,
            self.df.loc[range(start, end), 'Volume']
                .values / MAX_NUM_SHARES,
        ])

        front_padding = LOOKBACK_WINDOW_SIZE - self.current_step if start == 0 else 0
        end_padding = LOOKFORWARD_WINDOW_SIZE + self.current_step - self.df.shape[0] + 1 if end >= self.df.shape[0] else 0

        state = np.pad(state, [(0, 0), (front_padding, end_padding)], "constant", constant_values=0)

        # Append additional data and scale each value to between 0-1
        # The x axis is  the date. The y axis is the stock features
        obs = np.append(state, [
            [self.balance / MAX_ACCOUNT_BALANCE],
            [self.max_net_worth / MAX_ACCOUNT_BALANCE],
            [self.shares_held / MAX_NUM_SHARES],
            [self.cost_basis / MAX_SHARE_PRICE],
            [self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)],
        ], axis=1)

        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1.5:

            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            if self.shares_held + shares_bought != 0:
                # At the beginning, it's possible for the sum to be zero
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

            if shares_bought > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_bought, 'total': additional_cost,
                                    'type': "buy"})

        elif action_type <= 3:

            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

            if shares_sold > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': shares_sold, 'total': shares_sold * current_price,
                                    'type': "sell"})

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier + self.current_step
        done = self.net_worth <= 0 or self.current_step >= len(
            self.df.loc[:, 'Open'].values)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_step = 0
        self.trades = []

        return self._next_observation()

    def _render_to_file(self, filename='render_LB_10_FB_5.txt'):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        file = open(filename, 'a+')

        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})\n')
        file.write(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})\n')
        file.write(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})\n')
        file.write(f'Profit: {profit}\n\n')

        file.close()

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if mode == 'file':

            self._render_to_file(kwargs.get('filename', 'render.txt'))

        elif mode == 'live':
            if self.visualization == None:
                self.visualization = StockTradingGraph(
                    self.df, kwargs.get('title', None))

            if self.current_step > LOOKBACK_WINDOW_SIZE:
                self.visualization.render(
                    self.current_step, self.net_worth, self.trades, window_size=LOOKBACK_WINDOW_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
