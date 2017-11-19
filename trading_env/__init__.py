from .envs import available_envs_module

def available_envs():
    available_envs = [env_module.__name__.split('.')[-1] for env_module in available_envs_module]
    wip = ['realtime-v0']
    return available_envs

def make(env_id, obs_data_len, step_len,
         df, fee, deal_col_name='price', 
         feature_names=['price', 'volume'], 
         *args , **kwargs):
    envs = available_envs()
    assert env_id in envs , "env_id: {} not exist. try one of {}".format(env_id, str(envs).strip('[]'))
    assert deal_col_name in df.columns, "deal_col not in Dataframe please define the correct column name of which column want to calculate the profit."
    assert 'serial_number' in df.columns, "need serial_number columns to know where the day start."
    for col in feature_names:
        assert col in df.columns, "feature name: {} not in Dataframe.".format(col)
    #available_envs
    trading_env = available_envs_module[envs.index(env_id)].trading_env
    env = trading_env(env_id=env_id,
                      obs_data_len=obs_data_len, 
                      step_len=step_len ,df=df, fee=fee, 
                      deal_col_name=deal_col_name, 
                      feature_names=feature_names,
                      *args, **kwargs)
    return env    
