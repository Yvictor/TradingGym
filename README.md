# TradingGym

TradingGym is a toolkit for training and backtesting the reinforcement learning algorithms. This was inspired by OpenAI Gym and imitated the framework form. Not only traning env but also has backtesting and in the future will implement realtime trading env with Interactivate Broker API and so on.

This training env originally design for tickdata, but also support for ohlc data format. WIP.

### Installation
```
git clone https://github.com/Yvictor/TradingGym.git
cd TradingGym
python steup.py install
```

### Getting Started
``` python
import random
import trading_env

env = trading_env(obs_data_len=256, step_len=128,
                  df=df, fee=1, max_position=5, deal_col_name='price', 
                  feature_names=['price', 'volume'], fluc_div=100.0)

env.reset()
env.render()

state, reward, done, info = env.step(random.randrange(3))
```
- obs_data_len: observation data length
- step_len: when call step rolling windows will + step_len
- df exmaple
>|index|datetime|bid|ask|price|volume|serial_number|dealin|
>|-----|--------|---|---|-----|------|-------------|------|
>|0|2010-05-25 08:45:00|7188.0|7188.0|7188.0|527.0|0.0|0.0|
>|1|2010-05-25 08:45:00|7188.0|7189.0|7189.0|1.0|1.0|1.0|
>|2|2010-05-25 08:45:00|7188.0|7189.0|7188.0|1.0|2.0|-1.0|
>|3|2010-05-25 08:45:00|7188.0|7189.0|7188.0|4.0|3.0|-1.0|
>|4|2010-05-25 08:45:00|7188.0|7189.0|7188.0|2.0|4.0|-1.0|

- df: dataframe that contain data for trading 
> serial_number -> serial num of deal at each day recalculating

- fee: when each deal will pay the fee, set with your product. 
- max_position: the max market position for you trading share.
- deal_col_name: the column name for cucalate reward used.
- feature_names: list contain the feature columns to use in trading status.
