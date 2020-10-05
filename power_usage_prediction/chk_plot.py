# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:59:58 2020

@author: ISP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('D:\전력데이터 신서비스\쏘영\새 폴더')
#os.chdir('D:\전력데이터 신서비스\쏘영\AA5동')

folder_list = os.listdir()

for call_folder in range(len(folder_list)):
#call_folder = 4
    crt_folder = folder_list[call_folder]
    os.chdir(crt_folder)
    
    
    file_list = os.listdir()
    crt_file = file_list[0]
    data = pd.read_csv(crt_file)
    
    for i in range(1, len(file_list)):
        crt_file = pd.read_csv(file_list[i])
        data = pd.concat([data, crt_file], axis = 0)
        
    plt.plot( data.values[:,1] )
    plt.grid()
    plt.title(folder_list[call_folder])
    #plt.xticks(xticks=[i for i in range(0,len(data), 24)])
    plt.show()
    os.chdir('..')