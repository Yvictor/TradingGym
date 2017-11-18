import numpy as np
import pandas as pd
import trading_env

from datetime import datetime
st = datetime.now()
## need to refactor the testcase

df = pd.read_csv('trading_env/test/data/SGXTWsample.csv', index_col=0, parse_dates=['datetime'])

env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
                       df=df, fee=0.1, max_position=5, deal_col_name='Price', 
                       feature_names=['Price', 'Volume', 
                                      'Ask_price','Bid_price', 
                                      'Ask_deal_vol','Bid_deal_vol',
                                      'Bid/Ask_deal', 'Updown'], 
                       fluc_div=100.0)

env.reset()
print(env.df_sample['datetime'].iloc[0].date())
for i in range(500):
    #print(i)
    state, reward, done, info = env.step(np.random.randint(3))
    #print(state, reward)
    #env.render()
    if done:
        break
print(datetime.now() - st)