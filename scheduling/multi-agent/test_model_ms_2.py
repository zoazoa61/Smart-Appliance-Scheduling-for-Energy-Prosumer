import h5py
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
import tensorflow as tf
import Model_utils_3
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from util import *

# tf.compat.v1.global_variables
sns.set()
warnings.filterwarnings("ignore")
print(tf.test.gpu_device_name())

reward_save_dir = 'results/0817_2/'
DQN = tf.keras.models.load_model('models/multi_0817.hdf5')
DQN_ts = tf.keras.models.load_model('models/TS_multi_0817.hdf5')

#%%
start = '2018-04-01'
end = '2018-04-29'

# DQN hyperparameters
state_size = 6  # PV, load, SMP, past 24 hour average SMP, SOC, acc_load
action_size = 9
learning_rate = 0.001

# Training hyperparameters
episodes = 100
batch_size = 24
timesteps = 24*28  # 24시간 1일

# Exploration hyperparameters for epsilon greedy strategy
explore_start = 1.  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Q-learning hyperparameters
gamma = 0.999  # Discounting rate of future reward

# Memory hyperparameters
pretrain_length = 10000  # # of experiences stored in Memory during initialization
memory_size = 10000  # # of experiences Memory can keep

#%%
# path = 'Final Modified Data.csv'
path = 'state_true.csv'
df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
if start == None:
    start = df['date'][0]
if end == None:
    end = df['date'][-1]
date_range = pd.date_range(start, end, freq='H', closed='left')
df = df.loc[date_range[0]:date_range[-1],:]
del df['date']
df = df.reset_index(drop=True)

no_usage = 8  # 한달 8번 세탁기 사용
test_duration = 24*28  # 최근 1달 테스트
df = df[len(df) - test_duration:len(df)]
df = df.reset_index()
del df['index']

"""**Data Preprocessing_Standardization**"""
# MinMaxScaler
# The mean is not shifted to zero-centered

df_pv = df.iloc[:, 2:].values
df_load = df.iloc[:, 1:2].values
df_price = df.iloc[:, 0:1].values

# scale adjustment
df_pv = df_pv * 0.015
df_load = df_load * 0.05

sc_price = StandardScaler(with_mean=False)
sc_energy = StandardScaler(with_mean=False)

pv = sc_energy.fit_transform(df_pv)
load = sc_energy.transform(df_load)
price = sc_price.fit_transform(df_price)
# acc_load = np.cumsum(load)

x = np.concatenate([pv, load, price], axis=-1)

"""
**Hyperparameters Setting**
"""

appl = Appliance(action_size=1)

battery = Model_utils_3.Battery(action_size=action_size,
                                 scaler_energy=sc_energy,
                                 scaler_price=sc_price)
memory = Model_utils_3.Memory(memory_size)
memory_ts = Model_utils_3.Memory(memory_size)
appl = Appliance(action_size=1)
battery = Model_utils_3.Battery(action_size=action_size,
                                 scaler_energy=sc_energy,
                                 scaler_price=sc_price)

SOC = np.array([battery.initial_SOC])
historical_price = np.zeros(timesteps)
day = 0
hour = 0
timestep = 0
ts_stack = 0
done = False
pv_list = []
load_list = []
price_list = []
SOC_list = []
fault_list = []
ts_action_list = []
ts_reward_list = []
ns_action_list = []
ns_reward_list = []
x[:, 1] = load[:, 0]
av_price_list = []
trading_price_list = []
load_list.append(0)
while day < len(x) / 24:
    historical_price[timestep] = x[day * 24 + hour, 2]
    average_price = np.mean(np.array([price for price in historical_price if price != 0]))

    ts_state = np.concatenate(
        (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])), axis=-1)
    ts_action = np.argmax(DQN_ts.predict(np.expand_dims(ts_state, axis=0)))
    ts_stack, ts_load, ts_reward = appl.compute(ts_state, ts_action)
    x[day * 24 + hour, 1] += ts_load

    accumulated_load = np.cumsum(load_list)
    if hour == 0:
        state = np.concatenate(
            (x[day * 24 + hour, :], SOC, np.array([average_price]), np.array([accumulated_load[day * 24]])), axis=-1)
    else:
        state = np.concatenate(
            (x[day * 24 + hour, :], SOC, np.array([average_price]), np.array([0])), axis=-1)
    action = np.argmax(DQN.predict(np.expand_dims(state, axis=0)))

    next_SOC, ns_reward, state_update, comsumed_price, trading_price = battery.compute(state, action,
                                                                                       day * 24 + hour)  # 보상 얻는 텀
    trading_price_list.append(trading_price)

    pv_list.append(state_update[0])
    load_list.append(state_update[1])
    price_list.append(state_update[2])
    SOC_list.append(state_update[3])
    av_price_list.append(state_update[4])

    SOC = next_SOC
    ts_reward_list.append(ts_reward)
    ts_action_list.append(ts_action)
    ns_reward_list.append(ns_reward)
    ns_action_list.append(action)

    if hour < 23:
        hour += 1
        timestep += 1
        if timestep >= timesteps:  # timesteps=24
            timestep = 0

        historical_price[timestep] = x[day * 24 + hour, 2]
        average_price = np.mean(np.array([price for price in historical_price if price != 0]))
        ts_next_state = np.concatenate(
            (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])),
            axis=-1)
        # accumulated_load = np.sum(load_list, axis=0)
        next_state = np.concatenate(
            (x[day * 24 + hour, :], next_SOC, np.array([average_price]), np.array([0])),
            axis=-1)
    else:
        done = True
        day += 1
        hour = 0
        timestep += 1
        if timestep >= timesteps:
            timestep = 0
        if day < len(x) / 24:
            historical_price[timestep] = x[day * 24 + hour, 2]
            average_price = np.mean(np.array([price for price in historical_price if price != 0]))
            ts_next_state = np.concatenate(
                (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])),
                axis=-1)
            # accumulated_load = np.sum(load_list, axis=0)
            accumulated_load = np.cumsum(load_list)
            next_state = np.concatenate(
                (x[day * 24 + hour, :], next_SOC, np.array([average_price]), np.array([accumulated_load[day * 24]])),
                axis=-1)
        else:
            break


#%% plotting
def Tax_fn_true(acc_load): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    thres_q1 =200 #[W/day]
    thres_q2 = 400
    price_sum = 0
    if acc_load < thres_q1:
        price_sum +=acc_load * 93.3
    elif acc_load < thres_q1 + thres_q2:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1)* 187.9
    else:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1) * 187.9
        price_sum += (acc_load - thres_q1 - thres_q2) * 280.6
    return price_sum

tax = Model_utils_3.Tax_fn_total(np.sum(load_list), sc_energy)
total_load = sc_energy.inverse_transform(load_list).sum()
total_load_true = np.sum(df_load) + 0.5 * 9
trading_price_all = np.sum(trading_price_list)
Tax_fn_true(total_load_true - df_pv.sum())

tax_true = Tax_fn_true(np.sum(df_load) + 0.5 * 9)


(np.abs(np.where(ts_action_list)[0] % 24 - 20) / 24 * 100).mean()


#%%
def Tax_fn_true(acc_load): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    thres_q1 =200 #[W/day]
    thres_q2 = 400
    price_sum = 0
    if acc_load < thres_q1:
        price_sum +=acc_load * 93.3
    elif acc_load < thres_q1 + thres_q2:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1)* 187.9
    else:
        price_sum += thres_q1 * 93.3
        price_sum += (acc_load - thres_q1) * 187.9
        price_sum += (acc_load - thres_q1 - thres_q2) * 280.6
    return price_sum


def evaluation(scaled_load_list, trading_price_list, raw_load, raw_smp, raw_pv, ts_flag =True, ts_num_usage=9):
    # load list를 inverse transform
    scheduled_load = sc_energy.inverse_transform(np.array(scaled_load_list)).sum() + ts_flag * 0.5 * ts_num_usage
    # 거래 이익
    trading_price = np.sum(trading_price_list)
    print('1. 스케줄링 한 결과 (acc_load 포함)')
    print('예상 세금: {:.2f}, 예상 판매수익: {:.2f}, 예상 지불:{:.2f}'.format(Tax_fn_true(scheduled_load),trading_price,Tax_fn_true(scheduled_load)-trading_price))
    print('===============================================================================')
    print('2. 스케쥴링 안 한 결과')
    # reference
    ref_load = raw_load.sum() + ts_num_usage * 0.5
    ref_trading_price = raw_smp.mean() * raw_pv.sum()
    print('예상 세금: {:.2f}, 예상 판매 수익 (평균 smp로 pv 팔았을 시): {:.2f}, '
          '예상 지불 1: {:.2f}, 예상 지불 2 (pv 팔지 않고 자가발전): {:.2f}'.format(Tax_fn_true(ref_load),ref_trading_price,
                                                                         Tax_fn_true(ref_load) - ref_trading_price,
                                                                         Tax_fn_true(ref_load - raw_pv.sum())))

evaluation(load_list, trading_price_list, df_load,df_price, df_pv, ts_flag = False)