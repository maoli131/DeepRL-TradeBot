from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.TradingEnv import TradingEnv
from env.TradingEnv import LOOKFORWARD_WINDOW_SIZE
from env.TradingEnv import LOOKBACK_WINDOW_SIZE

import pandas as pd

TOTAL_TIME_STEPS = 20000
DISPLAY_MODE = 'file'  # use mode = 'file' to output files instead of videos
NOTE = "Multi-Training-Env"

#read the historical stock data
name = input("Please enter the stock historical data file name: ")
asset_name = name[: name.index('_')]

df = pd.read_csv('./data/invest_data/' + name, index_col=0)
df = df.iloc[::-1].reset_index(drop=True)
df = df.sort_values('Date')

train_size = int(len(df) * 0.9)
train_df, test_df = df[0:train_size], df[train_size:len(df)]
test_df = test_df.reset_index(drop=True)

# manually split the date based on the date when price surge happened
low_value_train = train_df[: -484].reset_index(drop=True)
high_value_train = train_df[-484:].reset_index(drop=True)

# ARIMA Predictor
# from predictors.arima import ArimaPredictor
# arima_predictor = ArimaPredictor()
# arima_predictor.train(train_df=train_df, column='Close')
# fc = arima_predictor.predict(steps=15)
# print(fc)

# The algorithms require a vectorized environment to run
train_env_low = DummyVecEnv([lambda: TradingEnv(low_value_train)])
train_env_high = DummyVecEnv([lambda: TradingEnv(high_value_train)])
test_env = DummyVecEnv([lambda: TradingEnv(test_df)])

model_low = PPO2(MlpPolicy, train_env_low, verbose=1)
model_low.learn(total_timesteps=TOTAL_TIME_STEPS)
model_low.save(save_path="./saved_model/ppo_{}_{}_{}_low.pkl"
               .format(asset_name, TOTAL_TIME_STEPS, NOTE), cloudpickle=True)

model_high = PPO2(MlpPolicy, train_env_high, verbose=1)
model_high.learn(total_timesteps=TOTAL_TIME_STEPS)
model_high.save(save_path="./saved_model/ppo_{}_{}_{}_high.pkl"
                .format(asset_name, TOTAL_TIME_STEPS, NOTE), cloudpickle=True)

obs = train_env_low.reset()

# back testing on training data
done = False
while not done:
    action, _states = model_low.predict(obs)
    obs, rewards, done, info = train_env_low.step(action)
    train_env_low.render(title=name[:-13], mode=DISPLAY_MODE, filename='LB_{}_LF_{}_{}_{}_{}_train_low.txt'.
                         format(LOOKBACK_WINDOW_SIZE, LOOKFORWARD_WINDOW_SIZE, TOTAL_TIME_STEPS, asset_name, NOTE))

# back testing on training data

obs = train_env_high.reset()
done = False
while not done:
    action, _states = model_high.predict(obs)
    obs, rewards, done, info = train_env_high.step(action)
    train_env_high.render(title=name[:-13], mode=DISPLAY_MODE, filename='LB_{}_LF_{}_{}_{}_{}_train_high.txt'.
                          format(LOOKBACK_WINDOW_SIZE, LOOKFORWARD_WINDOW_SIZE, TOTAL_TIME_STEPS, asset_name, NOTE))

done = False
model_low.set_env(test_env)
model_high.set_env(test_env)
obs = test_env.reset()

# back testing on testing data
rewards_going_down = 5
old_rewards = float('-inf')
model = model_low # initialize the model with model_low
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)

    if rewards < old_rewards:
        rewards_going_down -= 1
        if rewards_going_down < 0:
            if model is model_low:
                model = model_high
            else:
                model = model_low

    old_rewards = rewards
    test_env.render(title=name[:-13], mode=DISPLAY_MODE, filename='LB_{}_LF_{}_{}_{}_{}_test.txt'.
                    format(LOOKBACK_WINDOW_SIZE, LOOKFORWARD_WINDOW_SIZE, TOTAL_TIME_STEPS, asset_name, NOTE))

