import os
import logging

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class trading_env:
    def __init__(self, env_id, obs_data_len, step_len,
                 df, fee, max_position=5, deal_col_name='price', 
                 feature_names=['price', 'volume'], 
                 fluc_div=100.0, gameover_limit=5,
                 *args, **kwargs):
        """
        #assert df 
        # need deal price as essential and specified the df format
        # obs_data_leng -> observation data length
        # step_len -> when call step rolling windows will + step_len
        # df -> dataframe that contain data for trading(format as...)
            # price 
            # datetime
            # serial_number -> serial num of deal at each day recalculating
            
        # fee -> when each deal will pay the fee, set with your product
        # max_position -> the max market position for you trading share 
        # deal_col_name -> the column name for cucalate reward used.
        # feature_names -> list contain the feature columns to use in trading status.
        # ?day trade option set as default if don't use this need modify
        """
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        #self.file_loc_path = os.environ.get('FILEPATH', '')
        
        self.df = df
        self.action_space = np.array([3,])
        self.gym_actions = range(3)
        
        self.obs_len = obs_data_len
        self.feature_len = len(feature_names)
        self.observation_space = np.array([self.obs_len*self.feature_len,])
        self.using_feature = feature_names
        self.price_name = deal_col_name
        
        self.step_len = step_len
        self.fee = fee
        self.max_position = max_position
        
        self.fluc_div = fluc_div
        self.gameover = gameover_limit
        
        self.begin_fs = self.df[self.df['serial_number']==0]
        self.date_leng = len(self.begin_fs)
        
        self.render_on = 0
        self.buy_color, self.sell_color = (1, 2)
        self.new_rotation, self.cover_rotation = (1, 2)
        self.transaction_details = pd.DataFrame()
        self.logger.info('Making new env: {}'.format(env_id))
        
    def reset(self):
        random_int = np.random.randint(self.date_leng)
        if random_int == self.date_leng - 1:
            begin_point = self.begin_fs.index[random_int]
            end_point = None
        else:
            begin_point, end_point = self.begin_fs.index[random_int: random_int+2]
        self.df_sample = self.df.iloc[begin_point: end_point]
        self.step_st = 0
        self.price = self.df_sample[self.price_name].as_matrix()
        self.obs_features = self.df_sample[self.using_feature].as_matrix()
        self.obs_res = self.obs_features[self.step_st: self.step_st+self.obs_len]
        
        #maybe make market position feature in final feature, set as option
        self.posi_l = [0]*self.obs_len
        # self.position_feature = np.array(self.posi_l[self.step_st:self.step_st+self.obs_len])/(self.max_position*2)+0.5
        
        self.reward_sum = 0
        self.reward_fluctuant = 0
        self.reward_ret = 0
        self.transaction_details = pd.DataFrame()
        self.reward_curve = []
        self.t_index = 0
        
        self.buy_color, self.sell_color = (1, 2)
        self.new_rotation, self.cover_rotation = (1, 2)
        return self.obs_res
    
    
    def step(self, action):
        #price_current can be change next one to some avg of next_N or simulate with slippage
        next_index = self.step_st+self.obs_len+1
        self.price_current = self.price[next_index]
        self.make_real = 0
        reward = 0.0
        if action == 1 and self.max_position > self.posi_l[-1] >= 0:
            self.buy_price = self.price_current
            if self.posi_l[-1] > 0:
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                abs_pos = abs(self.position_share)
                self.reward_fluctuant = self.price_current*self.position_share - self.transaction_details.iloc[-1]['price_mean']*self.position_share - self.fee*abs_pos
                self.price_mean = (self.transaction_details.iloc[-1]['price_mean']*self.position_share + self.buy_price)/(self.position_share+1.0)
                self.position_share += 1
            else:
                self.reward_fluctuant = 0.0
                self.position_share = 1.0
                self.price_mean = self.buy_price
            self.posi_l += ([self.posi_l[-1] + 1 ]*self.step_len)
            self.t_index += 1
            transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact': 'Buy',
                                       'transact_type': 'new',
                                       'price': self.buy_price, 'share': 1,
                                       'price_mean': self.price_mean, 'position': self.position_share,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward': reward, 'reward_sum': self.reward_sum,
                                       'color': self.buy_color, 'rotation': self.new_rotation}
             ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
            self.transaction_details = pd.concat([self.transaction_details,transact_n])
        
        elif action == 2 and -self.max_position < self.posi_l[-1] <= 0:
            self.sell_price = self.price_current
            if self.posi_l[-1] < 0:
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                abs_pos = abs(self.position_share)
                self.reward_fluctuant = self.price_current*self.position_share - self.transaction_details.iloc[-1]['price_mean']*self.position_share  - self.fee*abs_pos
                self.price_mean = (-self.transaction_details.iloc[-1]['price_mean']*self.position_share + self.sell_price)/-(self.position_share-1.0)
                self.position_share -= 1
            else:
                self.reward_fluctuant = 0.0
                self.position_share = -1.0
                self.price_mean = self.sell_price
            
            self.posi_l+=([self.posi_l[-1] - 1 ]*self.step_len)
            self.t_index += 1
            transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact': 'Sell',
                                       'transact_type': 'new',
                                       'price': self.sell_price, 'share':-1,
                                       'price_mean': self.price_mean, 'position': self.position_share,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward': reward, 'reward_sum': self.reward_sum,
                                       'color': self.sell_color,'rotation': self.new_rotation}
             ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
            self.transaction_details = pd.concat([self.transaction_details,transact_n])        
            
        elif action == 1 and self.posi_l[-1]<0:
            self.buy_price = self.price_current

            self.position_share = self.transaction_details.iloc[-1].loc['position']
            abs_pos = abs(self.position_share)
            self.reward_fluctuant = self.price_current*self.position_share - self.transaction_details.iloc[-1]['price_mean']*self.position_share - self.fee*abs_pos
            self.position_share +=1
            
            reward = self.transaction_details.iloc[-1]['price_mean'] - self.buy_price - self.fee
            self.reward_sum += reward
            self.make_real = 1
            self.posi_l += ([self.posi_l[-1] + 1 ]*self.step_len)
            self.t_index += 1
            transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Buy',
                                       'transact_type':'cover',
                                       'price':self.buy_price,'share':1,
                                       'price_mean':self.price_mean, 'position':self.position_share,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.buy_color,'rotation':self.cover_rotation}
             ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
            self.transaction_details = pd.concat([self.transaction_details,transact_n])
        
        elif action == 2 and self.posi_l[-1]>0:
            self.sell_price = self.price_current
            
            self.position_share = self.transaction_details.iloc[-1].loc['position']
            abs_pos = abs(self.position_share)
            self.reward_fluctuant = self.price_current*self.position_share - self.transaction_details.iloc[-1]['price_mean']*self.position_share - self.fee*abs_pos
            self.position_share -=1
            
            reward = self.sell_price - self.transaction_details.iloc[-1]['price_mean'] - self.fee
            self.reward_sum += reward
            self.make_real = 1
            self.posi_l+=([self.posi_l[-1] - 1 ]*self.step_len)
            self.t_index +=1
            transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Sell',
                                       'transact_type':'cover',
                                       'price':self.sell_price,'share':-1,
                                       'price_mean':self.price_mean, 'position':self.position_share,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.sell_color,'rotation':self.cover_rotation}
             ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
            self.transaction_details = pd.concat([self.transaction_details,transact_n])
        
        elif action ==1 and self.posi_l[-1]==self.max_position:
            action = 0
        elif action == 2 and self.posi_l[-1]== -self.max_position:
            action = 0

        if action ==0:
            if self.posi_l[-1] != 0:
                self.posi_l+=([self.posi_l[-1]]*self.step_len)
                self.t_index +=1
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                abs_pos = abs(self.position_share)
                self.reward_fluctuant = self.price_current*self.position_share - self.transaction_details.iloc[-1]['price_mean']*self.position_share - self.fee*abs_pos
            else:
                self.posi_l+=([self.posi_l[-1]]*self.step_len)
                self.t_index +=1
                self.reward_fluctuant = 0.0
            
        self.reward_curve.append((self.step_st+self.obs_len, self.reward_fluctuant+self.reward_sum))
        
        
        self.step_st += self.step_len
        done = False
        if self.step_st+self.obs_len+self.step_len >= len(self.price):
            done = True
            if self.posi_l[-1] < 0:
                self.make_real = 1
                self.buy_price = self.price_current
                self.posi_l+=([0]*self.step_len)
                self.t_index += 1
                self.reward_fluctuant = 0.0
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                reward = (self.transaction_details.iloc[-1]['price_mean'] - self.buy_price - self.fee)*(-self.position_share)
                self.reward_sum +=reward
                transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Buy',
                                       'transact_type':'cover',
                                       'price':self.buy_price,'share':-self.position_share,
                                       'price_mean':self.price_mean, 'position':0,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.buy_color,'rotation':self.cover_rotation}
                ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
                self.transaction_details = pd.concat([self.transaction_details,transact_n])

                
            if self.posi_l[-1] > 0:
                self.make_real = 1
                self.sell_price = self.price_current
                self.posi_l+=([0]*self.step_len)
                self.t_index += 1
                self.reward_fluctuant = 0.0
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                reward = (self.sell_price - self.transaction_details.iloc[-1]['price_mean'] - self.fee)*self.position_share
                self.reward_sum +=reward
                transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Sell',
                                       'transact_type':'cover',
                                       'price':self.sell_price,'share':-self.position_share,
                                       'price_mean':self.price_mean, 'position':0,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.sell_color,'rotation':self.cover_rotation}
                ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
                self.transaction_details = pd.concat([self.transaction_details,transact_n])

                
                
        elif self.gameover and self.reward_sum+self.reward_fluctuant < -self.gameover:#3.5:
            done = True
            if self.posi_l[-1] < 0:
                self.make_real = 1
                self.buy_price = self.price_current
                self.posi_l+=([0]*self.step_len)
                self.t_index += 1
                self.reward_fluctuant = 0.0
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                reward = (self.transaction_details.iloc[-1]['price_mean'] - self.buy_price - self.fee)*(-self.position_share)
                self.reward_sum +=reward
                transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Buy',
                                       'transact_type':'cover',
                                       'price':self.buy_price,'share':-self.position_share,
                                       'price_mean':self.price_mean, 'position':0,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.buy_color,'rotation':self.cover_rotation}
                ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
                self.transaction_details = pd.concat([self.transaction_details,transact_n])

                
            if self.posi_l[-1] > 0:
                self.make_real = 1
                self.sell_price = self.price_current
                self.posi_l+=([0]*self.step_len)
                self.t_index += 1
                self.reward_fluctuant = 0.0
                self.position_share = self.transaction_details.iloc[-1].loc['position']
                reward = (self.sell_price - self.transaction_details.iloc[-1]['price_mean'] - self.fee)*self.position_share
                self.reward_sum +=reward
                transact_n = pd.DataFrame({'step': next_index,
                                       'datetime': self.df_sample.iloc[next_index].datetime,
                                       'transact':'Sell',
                                       'transact_type':'cover',
                                       'price':self.sell_price,'share':-self.position_share,
                                       'price_mean':self.price_mean, 'position':0,
                                       'reward_fluc': self.reward_fluctuant,
                                       'reward':reward,'reward_sum':self.reward_sum,
                                       'color':self.sell_color,'rotation':self.cover_rotation}
                ,index=[self.t_index],columns=['step','datetime','transact','transact_type','price','share','price_mean','position','reward_fluc',
                                            'reward','reward_sum','color','rotation'])
                self.transaction_details = pd.concat([self.transaction_details,transact_n])

        
        #self.logger.debug('Setp %d : make action %d'%(self.t_index,action))
        self.obs_res = self.obs_features[self.step_st: self.step_st+self.obs_len]
        
        # position feature
        #self.position_feature = np.array(self.posi_l[self.step_st:self.step_st+self.obs_len])/(self.max_position*2)+0.5
        #self.obs_res = np.array([self.price_feature,self.up_down_feature,self.ask_bid_feature,self.vol_feature,self.position_feature]).reshape(1,self.obs_len,self.feature_len)
        #self.obs_pv = np.concatenate([self.price_feature,self.up_down_feature,self.ask_bid_feature,self.vol_feature])
        self.reward_ret = 0.0
        if self.make_real == 0:
            self.reward_ret = self.reward_fluctuant/self.fluc_div
        elif self.make_real ==1:
            self.reward_ret = reward
        #if self.reward_ret < 0:
            #self.reward_ret = -0.005
        #self.obs_res = np.concatenate((self.obs_pv.reshape(self.obs_len*self.feature_len),self.position))#.astype(float)
        info = None
        return self.obs_res, self.reward_ret, done, info
    
    def render(self, save=False):
        if self.render_on == 0:
            matplotlib.style.use('dark_background')
            self.render_on = 1

            left, width = 0.1, 0.8
            rect1 = [left, 0.4, width, 0.55]
            rect2 = [left, 0.2, width, 0.2]
            rect3 = [left, 0.05, width, 0.15]

            self.fig = plt.figure(figsize=(15,8))
            self.fig.suptitle('%s'%self.df_sample['datetime'].iloc[0].date(), fontsize=14, fontweight='bold')
            #self.ax = self.fig.add_subplot(1,1,1)
            self.ax = self.fig.add_axes(rect1)  # left, bottom, width, height
            self.ax2 = self.fig.add_axes(rect2, sharex=self.ax)
            self.ax3 = self.fig.add_axes(rect3, sharex=self.ax)
            #fig, ax = plt.subplots()
            price_x = list(range(len(self.price[:self.step_st+self.obs_len])))
            self.price_plot = self.ax.plot(price_x, self.price[:self.step_st+self.obs_len], 'dodgerblue',zorder=1)
            self.vol_plot = self.ax3.plot(price_x, self.obs_features[:self.step_st+self.obs_len, 1], 'cyan')
            rect_high = self.price[self.step_st:self.step_st+self.obs_len].max() - self.price[self.step_st:self.step_st+self.obs_len].min()
            self.target_box = self.ax.add_patch(
                              patches.Rectangle(
                              (self.step_st, self.price[self.step_st:self.step_st+self.obs_len].min()),self.obs_len,rect_high,
                              label='observation',edgecolor=(1,1,1),facecolor=(0.95,1,0.1,0.8),linestyle=':',linewidth=2,
                              fill=True)
                              )     # remove background)
            self.fluc_reward_plot = self.ax2.fill_between([x[0] for x in self.reward_curve],0,[y[1] for y in self.reward_curve],facecolor='yellow',alpha=0.8)
            if len(self.transaction_details)!=0:
                self.reward_plot = self.ax2.fill_between(self.transaction_details.step,0,self.transaction_details.reward_sum,facecolor='cyan', alpha=0.5)
                self.share_plot = self.ax2.fill_between(self.transaction_details.step,0,self.transaction_details.position,facecolor='r', alpha=0.5)
                buy_record = self.transaction_details[self.transaction_details['transact']=='Buy']
                if len(buy_record)!=0:
                    trade_x = buy_record.step
                    trade_y = [self.price[i] for i in trade_x]
                    trade_color = [(1,0,0) if i =='new' else (1,0.7,0.7) for i in buy_record.transact_type]
                #trade_marker = ['v' if i =='Sell' else '^' for i in self.transaction_details.transact]
                    self.trade_plot = self.ax.scatter(x=trade_x,y=trade_y,s=100,marker='^',c=trade_color,edgecolors='none', zorder=2)
                sell_record = self.transaction_details[self.transaction_details['transact']=='Sell']
                if len(sell_record)!=0:
                    trade_x = sell_record.step
                    trade_y = [self.price[i] for i in trade_x]
                    trade_color = [(0,1,0) if i =='new' else (0.7,1,0.7) for i in sell_record.transact_type]
                    self.trade_plot = self.ax.scatter(x=trade_x,y=trade_y,s=100,marker='v',c=trade_color,edgecolors='none',zorder=2)

            self.ax.set_xlim(0,len(self.price[:self.step_st+self.obs_len])+200)
            plt.ion()
            #self.fig.tight_layout()
            plt.show()
            if save:
                self.fig.savefig('fig/%s.png' % str(self.t_index))
        elif self.render_on == 1:
            self.ax.lines.remove(self.price_plot[0])
            self.ax3.lines.remove(self.vol_plot[0])
            price_x = list(range(len(self.price[:self.step_st+self.obs_len])))
            self.price_plot = self.ax.plot(price_x, self.price[:self.step_st+self.obs_len], 'dodgerblue',zorder=1)
            self.vol_plot = self.ax3.plot(price_x, self.obs_features[:self.step_st+self.obs_len, 1], 'cyan')
            self.fluc_reward_plot.remove()
            self.target_box.remove()
            try:
                self.reward_plot.remove()
                self.share_plot.remove()
            except:
                pass
            self.fluc_reward_plot = self.ax2.fill_between([x[0] for x in self.reward_curve],0,[y[1] for y in self.reward_curve],facecolor='yellow',alpha=0.8)
            rect_high = self.price[self.step_st:self.step_st+self.obs_len].max() - self.price[self.step_st:self.step_st+self.obs_len].min()
            self.target_box = self.ax.add_patch(
                              patches.Rectangle(
                              (self.step_st, self.price[self.step_st:self.step_st+self.obs_len].min()),self.obs_len,rect_high,
                              label='observation',edgecolor=(1,1,1),facecolor=(0.95,1,0.1,0.75),linestyle=':',linewidth=2,
                              fill=True)
                              )              

            if len(self.transaction_details)!=0:
                try:
                    self.trade_plot.remove()
                except:
                    pass
                self.reward_plot = self.ax2.fill_between(self.transaction_details.step,0,self.transaction_details.reward_sum,edgecolors='cyan',facecolor='cyan')
                self.share_plot = self.ax2.fill_between(self.transaction_details.step,0,self.transaction_details.position,facecolor='r', alpha=0.5)

                buy_record = self.transaction_details[self.transaction_details['transact']=='Buy']
                if len(buy_record)!=0:
                    trade_x = buy_record.step
                    trade_y = [self.price[i] for i in trade_x]
                    trade_color = [(0.8,0,0) if i =='new' else (1,0.7,0.7) for i in buy_record.transact_type]
                #trade_marker = ['v' if i =='Sell' else '^' for i in self.transaction_details.transact]
                    self.trade_plot = self.ax.scatter(x=trade_x,y=trade_y,s=100,marker='^',c=trade_color,edgecolors='none', zorder=2)
                sell_record = self.transaction_details[self.transaction_details['transact']=='Sell']
                if len(sell_record)!=0:
                    trade_x = sell_record.step
                    trade_y = [self.price[i] for i in trade_x]
                    trade_color = [(0,1,0) if i =='new' else (0.7,1,0.7) for i in sell_record.transact_type]
                    self.trade_plot = self.ax.scatter(x=trade_x,y=trade_y,s=100,marker='v',c=trade_color,edgecolors='none',zorder=2)
            self.ax.set_xlim(0,len(self.price[:self.step_st+self.obs_len])+200)
            if save:
                self.fig.savefig('fig/%s.png' % str(self.t_index))
            plt.pause(0.0001)
    
    
    def backtest(self):
        self.gameover = None
        self.df_sample = self.df
        self.step_st = 0
        self.price = self.df_sample[self.price_name].as_matrix()
        self.obs_features = self.df_sample[self.using_feature].as_matrix()
        self.obs_res = self.obs_features[self.step_st: self.step_st+self.obs_len]
        
        #maybe make market position feature in final feature, set as option
        self.posi_l = [0]*self.obs_len
        # self.position_feature = np.array(self.posi_l[self.step_st:self.step_st+self.obs_len])/(self.max_position*2)+0.5
        
        self.reward_sum = 0
        self.reward_fluctuant = 0
        self.reward_ret = 0
        self.transaction_details = pd.DataFrame()
        self.reward_curve = []
        self.t_index = 0
        
        self.buy_color, self.sell_color = (1, 2)
        self.new_rotation, self.cover_rotation = (1, 2)
        return self.obs_res

    def show_pattern(self, transact_index):
        record_index = self.transaction_details.loc[transact_index]['step']
        return self.df_sample.iloc[record_index-self.obs_len-1:record_index]
    
    def show_future(self, transact_index):
        record_index = self.transaction_details.loc[transact_index]['step']
        nextdf = self.df_sample.iloc[record_index:]
        next_sess_index = nextdf[nextdf['serial_number']==0].iloc[0].name
        return nextdf.loc[:next_sess_index]