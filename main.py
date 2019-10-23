import gym
import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.TradingEnv import TradingEnv

import pandas as pd

#read the historical stock data
name = input("Please enter the stock historical data file name: ") 

df = pd.read_csv('./data/invest_data/' + name, index_col=0)
df = df.iloc[::-1].reset_index(drop=True)
df = df.sort_values('Date')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: TradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)

    obs, rewards, done, info = env.step(action)
    env.render(title=name[:-13]) # use mode = 'file' to output files instead of videos
