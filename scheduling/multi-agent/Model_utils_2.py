# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:50:06 2020

@author: user
"""

import h5py
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
tf.compat.v1.global_variables
# K.set_image_dim_ordering('th')
"""# Model

**Deep Q-Network Model**
"""

class DQNNet():
  
  def __init__(self, state_size, action_size, learning_rate):
    
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.model = self.create_model()
    
  def create_model(self):
    
    #state_size = (5, ) 
    input = Input(shape = (self.state_size, ))
    # input = Input(shape = self.state_size ,)
    # input= ([self.state_size,1])
    
    x = Dense(40, activation = "relu", 
              kernel_initializer = glorot_uniform(seed = 42))(input) #초기화 랜덤 seed 전달
    x = Dense(160, activation = "relu",
              kernel_initializer = glorot_uniform(seed = 42))(x)
    output = Dense(self.action_size, activation = "linear", 
              kernel_initializer = glorot_uniform(seed = 42))(x)
    
    model = Model(inputs = [input], outputs = [output])
    
    model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
    model.summary()
    
    return model


class DQNNet_ts():

    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        # state_size = (5, )
        input = Input(shape=(self.state_size,))
        # input = Input(shape = self.state_size ,)
        # input= ([self.state_size,1])

        x = Dense(40, activation="relu",
                  kernel_initializer=glorot_uniform(seed=42))(input)  # 초기화 랜덤 seed 전달
        x = Dense(160, activation="relu",
                  kernel_initializer=glorot_uniform(seed=42))(x)
        output = Dense(self.action_size, activation="linear",
                       kernel_initializer=glorot_uniform(seed=42))(x)

        model = Model(inputs=[input], outputs=[output])

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

"""**Memory Model**"""

# A tree based array containing priority of each experience for fast sampling

class SumTree():
  
  """
  __init__ - create data array storing experience and a tree based array storing priority
  add - store new experience in data array and update tree with new priority
  update - update tree and propagate the change through the tree
  get_leaf - find the final nodes with a given priority value
  """

  data_pointer = 0

  def __init__(self, capacity):
    
    """
    capacity - Number of final nodes containing experience
    data - array containing experience (with pointers to Python objects)
    tree - a tree shape array containing priority of each experience

     tree:
        0
       / \
      0   0
     / \ / \
    0  0 0  0 
    """
    self.capacity = capacity
    self.data = np.zeros(capacity, dtype = object)
    self.tree = np.zeros(2 * capacity - 1)

  def add(self, priority, data):
    
    # Start from first leaf node of the most bottom layer
    tree_index = self.data_pointer + self.capacity - 1

    self.data[self.data_pointer] = data # Update data frame
    self.update(tree_index, priority) # Update priority

    # Overwrite if exceed memory capacity
    self.data_pointer += 1
    if self.data_pointer >= self.capacity:  
      self.data_pointer = 0

  def update(self, tree_index, priority):

    # Change = new priority score - former priority score
    change = priority - self.tree[tree_index] 
    self.tree[tree_index] = priority

    # Propagate the change through tree
    while tree_index != 0: 
      tree_index = (tree_index - 1) // 2
      self.tree[tree_index] += change

  def get_leaf(self, v):

    parent_index = 0

    while True:
      left_child_index = 2 * parent_index + 1
      right_child_index = left_child_index + 1
      # Downward search, always search for a higher priority node till the last layer
      if left_child_index >= len(self.tree):
        leaf_index = parent_index
        break
      else: 
        if v <= self.tree[left_child_index]:
          parent_index = left_child_index
        else:
          v -= self.tree[left_child_index]
          parent_index = right_child_index

    data_index = leaf_index - self.capacity + 1
      
    # tree leaf index, priority, experience
    return leaf_index, self.tree[leaf_index], self.data[data_index]


class Memory():  # stored as (state, action, reward, updated_state) in SumTree

  
  PER_e = 0.01 
  PER_a = 0.6
  PER_b = 0.4
  PER_b_increment_per_sampling = 0.01
  absolute_error_upper = 1.  # Clipped abs error

  def __init__(self, capacity):
    
    self.tree = SumTree(capacity)

  def store(self, experience):
    
    # Find the max priority
    max_priority = np.max(self.tree.tree[-self.tree.capacity:])

    # If the max priority = 0, this experience will never have a chance to be selected
    # So a minimum priority is assigned
    if max_priority == 0:
      max_priority = self.absolute_error_upper

    self.tree.add(max_priority, experience)

  def sample(self, n):

    b_memory = []
    b_idx = np.empty((n, ))
    b_ISWeights =  np.empty((n, 1))

    priority_segment = self.tree.tree[0] / n   
    
    self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
    
    prob_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.tree[0]
    max_weight = (prob_min * n) ** (-self.PER_b)

    for i in range(n):
      a = priority_segment * i
      b = priority_segment * (i + 1)
      value = np.random.uniform(a, b)
      index, priority, data = self.tree.get_leaf(value)
      prob = priority / self.tree.tree[0]
      b_ISWeights[i, 0] = (prob * n) ** (-self.PER_b) / max_weight               
      b_idx[i]= index
      b_memory.append([data])
    return b_idx, b_memory, b_ISWeights

  def batch_update(self, tree_idx, abs_errors):
      
    # To avoid 0 probability
    abs_errors += self.PER_e 
    clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
    ps = np.power(clipped_errors, self.PER_a)

    for ti, p in zip(tree_idx, ps):
      self.tree.update(ti, p)

"""**Battery Model**"""

class Battery():
    
  def __init__(self, action_size, scaler_energy, scaler_price):
    
    """
    P_rated - charge/ discharge rate: 충방전비율 (kW)
    E_rated - rated capacity: 정격용량 (kWh)
    C_E - energy capital cost: 에너지자본비용 ($/kWh)
    LC - life cycle: 수명
    eta - efficiency: 효율
    DOD - depth of discharge: 배출량
    wear_cost - wear & operation cost: 감가상각비 ($/kWh/operation)
    wear_cost = (C_E * E_rated) / (eta * E_rated * LC * DOD)
    a1, m1, m2_c, m2_d: multiplier for energy gain: 에너지 이득을 위한 승수
    """ 
    # sc_price = StandardScaler(with_mean = False)
    # sc_energy = StandardScaler(with_mean = False)

    # self.P_rated = scaler_energy.transform(np.array([[1000]]))[0] # pu
    # self.E_rated = scaler_energy.transform(np.array([[5000]]))[0] # pu
    self.scaler_energy =scaler_energy
    self.scaler_price = scaler_price

    self.P_rated = scaler_energy.transform(np.array([[1]]))[0] # pu
    self.E_rated = scaler_energy.transform(np.array([[5]]))[0] # pu

    self.C_E = scaler_price.transform(np.array([[171]]))[0] # pu
    self.LC = 4996 #수명
    self.eta = 1. #효율
    self.DOD = 1. #배출량
    self.wear_cost = self.C_E / self.eta / self.DOD / self.LC #감가상각비
    self.action_sell = np.linspace(-1, -0.25, num = action_size//2, endpoint = True)
    self.action_use = np.linspace(-1, -0.25, num = action_size//2, endpoint = True)
    self.action_set = np.concatenate((self.action_sell,self.action_use,np.array([0])))#아무것도 안하는거/방출하는거(판매,자가사용))
    self.initial_SOC = 0.
    self.target_SOC = 0.5 # Decide the backup energy required: 필요한 백업 에너지 결정

    self.a1 = 4.
    self.m1 = 2
    self.m2_c = 0.9
    self.m2_d = 1.2
    self.m2_d2 = 1
    # self.plot_multiplier()
    
  def compute(self, state, action, timestep):
    
    current_pv = state[0]
    current_load = state[1]
    current_price = state[2]
    current_SOC = state[3]
    average_price = state[4]
    # acc_load = state[5]

    pv_SOC = current_pv / self.E_rated
    next_SOC = np.minimum(1., current_SOC + pv_SOC)
    penalty = 0
    if current_SOC + pv_SOC > 0:
        penalty += -3

    delta_SOC = self.action_set[action] * next_SOC
    next_SOC += delta_SOC

    if action >= 4 and action <= 7:
        current_load = current_load + delta_SOC * self.E_rated

    # """ SOC penalty 2: 전기요금을 고려한 거래 """
    # 현재가가 평균가보다 비싼데 안팔았을경우 페널티
    if current_price > average_price * 1.3:
        if self.action_set[action] == 0:
            penalty += 1.5
        else:  # 현재가가 평균가보다 비싼경우 자가사용/판매
            if action < 4:
                penalty += -1.5
    elif current_price < average_price * 0.5: # 현재가가 평균가보다 싼경우 자가사용/판매
            penalty += -10
    elif current_price < average_price * 0.7: # 현재가가 평균가보다 싼경우 자가사용/판매
        if action >= 4 and action <= 7:
            penalty += -7
    elif current_price < average_price * 0.9: # 현재가가 평균가보다 싼경우 자가사용/판매
        if action >= 4 and action <= 7:
            penalty += -5

    # Compute piecewise multiplier
    # if next_SOC < self.target_SOC:  # Before target SOC is met
    #     multiplier = 1 + self.a1 * np.exp(-(-np.log((self.m1 - 1) / self.a1)) * \
    #                                       next_SOC / self.target_SOC)
    # else:  # After target SOC is met
    #     if delta_SOC >= 0:  # Charge
    #         multiplier = np.exp(-(-np.log(self.m2_c)) * (next_SOC - self.target_SOC) \
    #                             / (1 - self.target_SOC))
    #     else:  # Discharge
    #         multiplier = self.m2_d * np.exp(-(-np.log(self.m2_d2 / self.m2_d)) * \
    #                                         (next_SOC - self.target_SOC) \
    #                                         / (1 - self.target_SOC))

    # 누진제 reward 계산
    # acc_load = acc_load + current_load
    # consumed_price, tax_cost = Tax_fn(acc_load, timestep+1, self.scaler_energy, self.scaler_price)
    consumed_price, tax_cost = Tax_fn(current_load, timestep + 1, self.scaler_energy, self.scaler_price)

    # 배터리로 인한 에너지 이득
    energy_gain = average_price * (next_SOC - current_SOC) * self.E_rated
    if action < 4:# 판매했을경우만 거래비용 고려
        trading_cost = current_price * delta_SOC * self.E_rated
        trading_price = self.scaler_price.inverse_transform(np.array([current_price]).reshape(-1, 1)) \
                        * self.scaler_energy.inverse_transform(np.array([-1 * delta_SOC * self.E_rated]).reshape(-1, 1))  # 실제로 판 가격
        trading_price = trading_price[0,0]
    else:
        trading_cost = 0
        trading_price = 0
    wear_cost = self.wear_cost * np.abs((next_SOC - current_SOC) * self.E_rated)
    state[1] = current_load

    reward = energy_gain - trading_cost - wear_cost - tax_cost - penalty
    # print(f'{energy_gain}, - {trading_cost}, - {wear_cost}, - {tax_cost}')
    return next_SOC, reward[0], state, consumed_price, trading_price
  
  def plot_multiplier(self):
    
    SOC = np.linspace(0, 1, 100)
    multiplier_c = []
    multiplier_d = []
     
    # Compute piecewise multiplier
    for i in range(len(SOC)):
      if SOC[i] < self.target_SOC: # Before target SOC is met
        multiplier = 1 + self.a1 * np.exp(-(-np.log((self.m1 - 1) / self.a1)) * \
                                           SOC[i] / self.target_SOC)      
        multiplier_c.append(multiplier)
        multiplier_d.append(multiplier)
      else: # After target SOC is met
        multiplier = np.exp(-(-np.log(self.m2_c)) * (SOC[i] - self.target_SOC) \
                            / (1 - self.target_SOC))
        multiplier_c.append(multiplier)
        
        multiplier = self.m2_d * np.exp(-(-np.log(self.m2_d2 / self.m2_d)) * \
                                        (SOC[i] - self.target_SOC) \
                                        / (1 - self.target_SOC))
        multiplier_d.append(multiplier)
    
    plt.plot(SOC, np.array(multiplier_c), "r", label = "Charge")
    plt.plot(SOC, np.array(multiplier_d), "b", label = "Discharge")
    plt.xlabel("SOC")
    plt.ylabel("Multiplier")
    plt.legend()
    plt.show()


def Memory_Initialization(memory,battery,timesteps,pretrain_length,x,action_size, sc_energy):
    np.random.seed(42)
    # Memory initialization
    SOC = np.array([battery.initial_SOC])
    historical_price = np.zeros(timesteps) #timesteps = 24
    day = 0
    hour = 0
    timestep = 0
    done = False
    load_list = []
    load_list.append(0)
    for i in range(pretrain_length):
      
      # Keep track of the past 24 hours' electricity price: 지난 24시간의 전기요금 추적
      historical_price[timestep] = x[day * 24 + hour, 2] 
      average_price = np.mean(np.array([price for price in historical_price if price != 0]))

      accumulated_load = np.sum(load_list, axis=0)
      state = np.concatenate((x[day * 24 + hour, :], SOC, np.array([average_price])), axis = -1)
      action = np.random.randint(0, action_size)
      
      # Compute the reward and new state based on the selected action: 보상 얻는 텀
      next_SOC, reward, state_update, _, _ = battery.compute(state, action,day*24+hour)

      load_list.append(state_update[1])
      accumulated_load = np.sum(load_list, axis=0)

      # Store the experience in memory
      if hour < 23:
        hour += 1
        timestep += 1
        if timestep >= timesteps:
          timestep = 0
        historical_price[timestep] = x[day * 24 + hour, 2]
        average_price = np.mean(np.array([price for price in historical_price if price != 0]))
        next_state = np.concatenate((x[day * 24 + hour + 1, :], next_SOC, np.array([average_price])), axis = -1)
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
          next_state = np.concatenate((x[day * 24 + hour + 1, :], next_SOC, np.array([average_price])), axis = -1)
        else:
          break
          
      SOC = next_SOC
      experience = state, action, reward, next_state, done
      memory.store(experience)
      return(memory)


def TS_Memory_Initialization(memory, appl, timesteps, pretrain_length, x, action_size=2):
    np.random.seed(42)
    historical_price = np.zeros(timesteps)  # timesteps = 24
    day = 0
    hour = 0
    timestep = 0
    done = False
    ts_stack = 0
    # Memory initialization
    for i in range(pretrain_length):
        historical_price[timestep] = x[day * 24 + hour, 2]
        average_price = np.mean(np.array([price for price in historical_price if price != 0]))

        ts_state = np.concatenate(
            (x[day * 24 + hour, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])), axis=-1)
        ts_action = np.random.randint(0, action_size)

        # Compute the reward and new state based on the selected action: 보상 얻는 텀
        ts_stack, ts_load, ts_reward = appl.compute(ts_state, ts_action)

        # Store the experience in memory
        if hour < 23:
            hour += 1
            timestep += 1
            if timestep >= timesteps:
                timestep = 0
            historical_price[timestep] = x[day * 24 + hour, 2]
            average_price = np.mean(np.array([price for price in historical_price if price != 0]))
            ts_next_state = np.concatenate(
            (x[day * 24 + hour + 1, 2:], np.array([average_price]), np.array([ts_stack]), np.array([hour / 24])), axis=-1)
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
            else:
                break

            experience = ts_state, ts_action, ts_reward, ts_next_state, done
            memory.store(experience)
            return memory

import random
# import numpy as np
""" 특정시간만 선택했을 경우를 추가적으로 정해야함"""
def usage_appliance(no_usage, test_duration, load): #사용횟수, 테스트기간포인트수, 부하데이터  
    count = 1  
    time_interval = int(test_duration/no_usage) 
    min_interval = 3*24 #최소 3일 간격
    time_usage = np.zeros(test_duration)
    while count > 0:  
        usage_list = []
        for i in range(no_usage):        
            lst = list(np.arange(time_interval*i, time_interval*(i+1), 1))
            time_usage_idx = random.sample(lst, 1)
            time_usage_idx.sort()
            usage_list.append(time_usage_idx)
        usage_list = np.array(usage_list)
        
        diff_list = []
        for j in range(len(usage_list)-1):
            diff = usage_list[j+1] - usage_list[j]
            diff_list.append(diff)
        eval_diff = np.array(diff_list) < min_interval        
        if True not in eval_diff:
            count = 0           
    usage_list = list(usage_list[:,0])
    # time_usage[usage_list] = 0.5
    load[usage_list] = load[usage_list] + 0.5

    return(load)

def Tax_fn_sparse(load, timestep, sc_energy, sc_price, test_duration=24*28):
    hour = timestep % 24
    day = timestep // 24
    if day == 0:
        reward = 0
    else:
        acc_load = load[day * 24 -1]


def Tax_fn(load, timestep, sc_energy, sc_price, test_duration=24*28): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    load = load * test_duration

    thres_q1 = sc_energy.transform(np.array([200]).reshape(-1,1))[0,0] #[W/day]
    thres_q2 = sc_energy.transform(np.array([400]).reshape(-1,1))[0,0]

    price_sum = 0
    # applied_price_sum = 0
    if load < thres_q1:
        reward = -1.5
    elif load < thres_q1 + thres_q2:
        reward = 0
    else:
        reward = +1.5
    return (price_sum, reward)

def Tax_fn_total(acc_load, sc_energy): #unit[kW/h]
    # 한달이 끝났을 때 예상 load 계산
    thres_q1 = sc_energy.transform(np.array([200]).reshape(-1,1))[0,0] #[W/day]
    thres_q2 = sc_energy.transform(np.array([400]).reshape(-1,1))[0,0]
    price_sum = 0
    if acc_load < thres_q1:
        price_sum += inverse_transform(sc_energy, acc_load) * 93.3
    elif acc_load < thres_q1 + thres_q2:
        price_sum += inverse_transform(sc_energy, thres_q1) * 93.3
        price_sum += inverse_transform(sc_energy, (acc_load - thres_q1)) * 187.9
    else:
        price_sum += inverse_transform(sc_energy, thres_q1) * 93.3
        price_sum += inverse_transform(sc_energy, (acc_load - thres_q1)) * 187.9
        price_sum += inverse_transform(sc_energy, (acc_load - thres_q1 - thres_q2)) * 280.6
    return price_sum


def DQN_minibatch(memory, DQN, batch_size, cont, gamma, action_size):
    tree_idx, batch, ISWeights_mb = memory.sample(batch_size)  # Obtain random mini-batch from memory
    # chk_lst.append([tree_idx, batch, ISWeights_mb])
    # print(tree_idx, batch, ISWeights_mb )

    states_mb = np.array([each[0][0] for each in batch])
    actions_mb = np.array([each[0][1] for each in batch])
    rewards_mb = np.array([each[0][2] for each in batch])
    next_states_mb = np.array([each[0][3] for each in batch])
    dones_mb = np.array([each[0][4] for each in batch])

    target_batch = []
    if cont:
        targets_mb = DQN.predict(states_mb)
        q_next_state = DQN.predict(next_states_mb)
    else:
        targets_mb = DQN.model.predict(states_mb)
        # Update those targets at which actions are taken
        q_next_state = DQN.model.predict(next_states_mb)

    for i in range(0, len(batch)):
        action = np.argmax(q_next_state[i])
        if dones_mb[i] == 1:
            target_batch.append(rewards_mb[i])
        else:
            target = rewards_mb[i] + gamma * q_next_state[i][action]
            target_batch.append(rewards_mb[i])

    # Replace the original with the updated targets
    one_hot = np.zeros((len(batch), action_size))
    one_hot[np.arange(len(batch)), actions_mb] = 1
    targets_mb = targets_mb.astype("float64")
    target_batch = np.array([each for each in target_batch]).astype("float64")
    np.place(targets_mb, one_hot > 0, target_batch)

    absolute_errors = []
    if cont:
        loss = DQN.train_on_batch(states_mb, targets_mb, sample_weight=ISWeights_mb.ravel())
        predicts_mb = DQN.predict(states_mb)
    else:
        loss = DQN.model.train_on_batch(states_mb, targets_mb, sample_weight=ISWeights_mb.ravel())
        # Update priority
        predicts_mb = DQN.model.predict(states_mb)

    for i in range(0, len(batch)):
        absolute_errors.append(np.abs(predicts_mb[i][actions_mb[i]] - targets_mb[i][actions_mb[i]]))
    absolute_errors = np.array(absolute_errors)

    tree_idx = np.array([int(each) for each in tree_idx])
    memory.batch_update(tree_idx, absolute_errors)
    return memory

def inverse_transform(scaler, data):
    return scaler.inverse_transform(np.array([data]).reshape(-1, 1))[0,0]