#from ABMpy_region.model import ABM as ABM
from ABMpy_region.agent import ABM
import numpy as np
import pickle
import json
import os
import time

''' grid search '''
job_id = 0

# default params
params={
        'A0':122,'I0':100,
        'A2I':0.45,
        'max_mix':30,
        'nPop':10**6,
        'region':0
        }

# placeholder for vaccine plan, vaccination won't start in sim
params['vaccine_date']=1000 
params['vaccine_plan']={'total':20,'Pfizer':10,'J&J':10}

variables=['group_size','gamma','gamma_M','S2A']

# make grid
group_size=np.array([3,5,10,15,20,30,50])
gamma=np.array([0.3,0.5,0.7,0.9,0.95])
gamma_M=np.linspace(0.01,0.99,10)
S2A=np.linspace(0.006,0.02,18)

param=np.array(np.meshgrid(group_size,gamma,gamma_M,S2A)).T.reshape(-1,4)

# total jobs
jobs=6300
# chunk per CPU
chunk=4
# output prefix 
outpath='output_march23/model_job_'

j=0

end=min((job_id+1)*chunk,param.shape[0])

for p in param[job_id*chunk:end]:
    
    # set params
    param_dict={}
    for i in range(4):
        params[variables[i]]=p[i]
        param_dict[variables[i]]=p[i]
    params['group_size']=int(p[0]) 
    print('param',p)
    for seed in range(10):
        savefile=outpath+str(job_id)+'_'+str(j)+'_s'+str(seed)+'.pickle'
        if os.path.exists(savefile):
            continue
        np.random.seed(seed)
        # run sim
        m=ABM(params=params)
        t1=time.time()
        m.run(steps=450) 

        print('time ',np.round((time.time()-t1)/60,3),'mins')
        result={'params':param_dict,'curve':m.data}

        with open(savefile, 'wb') as fp:
            pickle.dump(result, fp)
        print('results saved')
    j+=1

       
