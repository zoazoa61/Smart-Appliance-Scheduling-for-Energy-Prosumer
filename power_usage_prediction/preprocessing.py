# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:59:58 2020

@author: ISP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('D:\전력데이터 신서비스\쏘영\preprocessing')
#os.chdir('D:\전력데이터 신서비스\쏘영\AA5동')

folder_list = os.listdir()

data_list = []
#for call_folder in range(len(folder_list)):
call_folder = 2
crt_folder = folder_list[call_folder]
os.chdir(crt_folder)


file_list = os.listdir()
crt_file = file_list[0]
data = pd.read_csv(crt_file)

for i in range(1, len(file_list)):
    crt_file = pd.read_csv(file_list[i])
    data = pd.concat([data, crt_file], axis = 0)
    #data_list.append(data)
    #os.chdir('..')

#data = pd.concat([data['Unnamed: 0'],data['0']],axis=1)
###########################################

""" Check null plot bar"""
null_data = []
null_point = data.isnull().values[:,1]
for i in range(len(data)):
    if null_point[i]:
        crt_date = data.values[i,0]
        crt_value = data.values[i,1]
    
#        null_data.append([crt_date[0:7], crt_value])
        null_data.append([crt_date, crt_value])
        
null_data = pd.DataFrame(null_data)
index, count = np.unique(null_data.values[:,0], return_counts=True)
null_index = pd.DataFrame(index)
#null_count = pd.DataFrame(count)
#null_arranged = pd.concat([null_index, null_count], axis=1)
#
#plt.bar(null_arranged.values[:,0], null_arranged.values[:,1])
#plt.xticks(rotation=45)
#plt.title(crt_folder)
#plt.show()    
#    
#is_null_date = data[null_point]
#for i in range(1, len(data)-1):
#    crt_date = data.values[i,0]
#    befor_value = data.values[i-1,1]
#    after_value = data.values[i-1,1]
#
#    if crt_date in is_null_date:
#        data.values[i,1] = ()

tmp_data = data['0']
tmp_data = tmp_data.values[0:len(tmp_data)]
#tmp_data = tmp_data.values[0:len(tmp_data)-585]#AA8호
#tmp_data = tmp_data.values[0:len(tmp_data)-563]#AA8호

tmp_date = data['Unnamed: 0']
tmp_date = tmp_date.values[0:len(tmp_data)]

count = 0
i = count
while count < len(tmp_data):
#while count < 70:

    tmp_list = []
    i = count
    
    tmp_value = tmp_data[i]
    if np.isnan(tmp_value):
        save_before = tmp_data[i-1]
        iteration = 0
        while np.isnan(tmp_value):
            count+=1
            iteration+=1
            tmp_value = tmp_data[count]
            
#            print(count)
        save_after = tmp_value
        tmp_list.append([save_before, save_after, iteration])
        inpolation = (save_after - save_before)/iteration
        
        for j in range(1,iteration+1):
            inpolation_value = save_after - inpolation*j
            tmp_data[count-j] = inpolation_value
#            print(inpolation_value)
        print(iteration)
        
    else:
        count+=1
    
tmp_data = pd.DataFrame(tmp_data)
tmp_date = pd.DataFrame(tmp_date)
data_processed = pd.concat([tmp_date, tmp_data], axis=1)

data_processed.plot(x=data_processed.values[:,0])
plt.xticks(rotation=45)
plt.title(crt_folder)
plt.show()    

#data_processed.to_csv(crt_folder+'.csv')
    
    
    