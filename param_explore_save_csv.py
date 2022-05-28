import pickle
import matplotlib.pyplot as plt
#import modin.pandas as pd
import pandas as pd
import numpy as np
import os

col=['file#','group_size','gamma','gamma_M','S2A']
col.extend([i for i in range(451)])
params=pd.DataFrame(columns=col)
#params=pd.read_csv("fit_data_march26.csv")
job=range(1575)
i=range(4)

l=[]
k=0
files=params['file#'].to_numpy()
for j in job:
    for ii in i:
        for seed in range(10):
            file='/home-4/yhuang98@jhu.edu/scratch/run_sim/output_march26/model_job_' \
                    +str(j)+'_'+str(ii)+'_s'+str(seed)+'.pickle'
            #print(file)
            if os.path.exists(file):
                print(k)
                with open(file,'rb') as fp:
                    m=pickle.load(fp)
                m['params']['file#']=k
                row=pd.Series(m['params'])
                #print('x')
                row=row.append(m['curve'].cum_case)
                l.append(row)
            k+=1
                #%time params=params.append(row, ignore_index=True)
params=pd.DataFrame(l) 
#params=params.append(params2,ignore_index=True) 
params.to_csv("fit_data_march30.csv", index=False)
