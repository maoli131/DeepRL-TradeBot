import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.TradingEnv import TradingEnv
from env.TradingEnv import LOOKFORWARD_WINDOW_SIZE
from env.TradingEnv import LOOKBACK_WINDOW_SIZE

import pandas as pd

TOTAL_TIME_STEPS = 20000
DISPLAY_MODE = 'file'  # use mode = 'file' to output files instead of videos


#read the historical stock data
name = input("Please enter the stock historical data file name: ")
asset_name = name[: name.index('_')]

df = pd.read_csv('./data/invest_data/' + name, index_col=0)
df = df.iloc[::-1].reset_index(drop=True)
df = df.sort_values('Date')

train_size = int(len(df) * 0.9)
train_df, test_df = df[0:train_size], df[train_size:len(df)]
test_df = test_df.reset_index(drop=True)

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
test_env = DummyVecEnv([lambda: TradingEnv(test_df)])

model = PPO2(MlpPolicy, train_env, verbose=1)
model.learn(total_timesteps=TOTAL_TIME_STEPS)
model.save(save_path="./saved_model/ppo_{}_{}.pkl".format(asset_name, TOTAL_TIME_STEPS), cloudpickle=True)

obs = train_env.reset()

# back testing on training data
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = train_env.step(action)
    train_env.render(title=name[:-13], mode=DISPLAY_MODE, filename='LB_{}_LF_{}_{}_{}_train.txt'.
                     format(LOOKBACK_WINDOW_SIZE, LOOKFORWARD_WINDOW_SIZE, TOTAL_TIME_STEPS, asset_name))

done = False
model.set_env(test_env)
obs = test_env.reset()

# back testing on testing data
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render(title=name[:-13], mode=DISPLAY_MODE, filename='LB_{}_LF_{}_{}_{}_test.txt'.
                    format(LOOKBACK_WINDOW_SIZE, LOOKFORWARD_WINDOW_SIZE, TOTAL_TIME_STEPS, asset_name))
