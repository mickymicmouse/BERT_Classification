# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:38:19 2020

@author: seungjun
"""

import numpy as np
import pandas as pd
import os

yolo_data=os.listdir(r"/home/itm1/seungjun/mixed_data/second_version")
for i in range(len(yolo_data)):
    if "_mix" in yolo_data[i]:
        mixed_data=os.listdir(r"/home/itm1/seungjun/mixed_data/second_version"+"/"+yolo_data[i])
        for j in mixed_data:
            if ".txt" in j:
                tab=pd.read_table(r"/home/itm1/seungjun/mixed_data/second_version"+"/"+yolo_data[i]+"/"+j,header=None, sep=" ")
                tab[0]=i+2
                f=open(r"/home/itm1/seungjun/mixed_data/second_version"+"/"+yolo_data[i]+"/"+j,'w')
                for k in range(0,4,1):
                    f.write(str(tab[k][0]))
                    f.write(" ")
                f.write(str(tab[4][0]))
                f.close()
        print(str(yolo_data[i])+" is finished at "+ str(i))


