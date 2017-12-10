from .envs import available_envs_module

def available_envs():
    available_envs = [env_module.__name__.split('.')[-1] for env_module in available_envs_module]
    wip = ['realtime-v0']
    return available_envs

def make(env_id, obs_data_len, step_len,
         df, fee, deal_col_name='price', 
         feature_names=['price', 'volume'], 
         return_transaction=False,
         *args , **kwargs):
    """
    v0: currently not maintain

    v1:
    # assert df 
    # need deal price as essential and specified the df format
    # obs_data_leng -> observation data length
    # step_len -> when call step rolling windows will + step_len
    # df -> dataframe that contain data for trading(format as...)
        # price 
        # datetime
        # serial_number -> serial num of deal at each day recalculating
        
    # fee : when each deal will pay the fee, set with your product
    # max_position : the max market position for you trading share 
    # deal_col_name : the column name for cucalate reward used.
    # feature_names : list contain the feature columns to use in trading status.
    # return_transaction : bool, list or dict default False
                        True will return all column include 'mkt_pos', 'mkt_pos_var', 'entry_cover', 'avg_hold_price', 'fluc_reward', 'make_real', 'reward'
                        use list to define which column to return
                        use dict to define which column to return and dict value as the func apply to array  
    # ?day trade option set as default if don't use this need modify
    """
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
                      return_transaction=return_transaction,
                      *args, **kwargs)
    return env    
