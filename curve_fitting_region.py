import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

''' find the best-fit sim for a region '''

# each job_id coresponds to a region job_id=range(0,number_of_regions)
job_id = 0

# read in sim data from grid search
sim=pd.read_csv("~/scratch/run_sim/fit_data_march30.csv")
col={str(i):i for i in np.arange(451)}
sim.rename(columns=col,inplace=True)

# read in actual data
data=pd.read_csv("prevalance_712_cumcase.csv")

# csv of regional info, includes fitting window in data, start/end date, offset in cases
df=pd.read_csv("fitting_info_aug3.csv")



def rmse(a1,a2):
    ''' rmse computation '''
    # weights
    w=(np.arange(1,a1.shape[0]+1))
    w[-15:]=w[-1]*500
    # quadratic
    w=w/w.sum()
    
    return np.sqrt((w*(a1-a2)**2).sum())

def make_data(r,start,end,offset):
    ''' get the row of region r from data, to be ready to be compared directly to sim '''
    data_r=[]
    row=data[data.location==r]
    for i in range(int(start),int(end)+1):
        data_r.append(row[str(int(i))].item())
    data_r=np.array(data_r)*10**6-offset
    #print('offset',offset)
    return data_r


i=job_id

r=df.loc[i,'location']
print(r)
start=df.loc[i,'start']
end=df.loc[i,'end']
offset=df.loc[i,'offset']
param_names=['group_size','gamma','gamma_M','S2A']
result={'location':r}

min_rmse=10**10

cases=make_data(r,start,end,offset)
days=np.arange(len(cases)) 

# loop through all sims from grid search
for i, row in sim.iterrows():
    # get sim prediction for specific day
    sim_pred=row[days].to_numpy()

    current_rmse=rmse(sim_pred,cases)

    # update minimal rsme 
    if current_rmse<min_rmse:
        min_rmse=current_rmse
        min_params=row[:4].to_numpy()
        min_row=row
        min_sim_number=i
    min_rmse=min(rmse(sim_pred,cases),min_rmse)


#print(param_names)
print(np.round(min_params,3))

# record best fit results
result['min_sim_number']=min_sim_number
result['min_rmse']=min_rmse
result['cum_case_in_1M']=cases[-1]
result['min_rmse_fraction']=min_rmse/cases[-1]

for p in range(4):
    result[param_names[p]]=min_params[p]

# saving results 
with open('data/fit_output_83/fit_'+str(job_id)+'.pkl','wb') as f:
    pickle.dump(result,f)