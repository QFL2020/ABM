import numpy as np
import json

''' default values '''
  
CASE_FATALITY=[0.00018,0.00327,0.10143,0.00025,0.00723,0.12524,0.000342,0.006213,0.192717,0.000475,0.013737,0.23795599999999997]

# DO NOT CHANGE:
SCALE_FACTOR_S2A=np.array([0.476, 0.976,1.0,0.5236000000000001,1.0736,1.1,0.5711999999999999,1.1712,1.2,0.6283200000000001,1.2883200000000001,1.32]) 
# setting female senior = 1

DEMO_DIST=np.array([0.12 , 0.26 , 0.12 , 0.13 , 0.265, 0.105, 0.   , 0.   , 0.   ,0.   , 0.   , 0.   ])

# 1 - vaccine efficacy
VACCINE_RISK_REDUCTION ={'Pfizer':0.05,'J&J':0.342}

'''
 sex=range(2) # ['F','M']
 age=range(3) #['Child','Adult','Senior']
 race=range(2) # ['reference','minority']
 
 demo = 
 array([[0, 0, 0],
       [0, 1, 0],
       [0, 2, 0],
       [1, 0, 0],
       [1, 1, 0],
       [1, 2, 0],
       [0, 0, 1],
       [0, 1, 1],
       [0, 2, 1],
       [1, 0, 1],
       [1, 1, 1],
       [1, 2, 1]])
'''