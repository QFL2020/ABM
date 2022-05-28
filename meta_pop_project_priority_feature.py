from ABMpy_region.model import Meta_ABM 
from ABMpy_region.agent import ABM
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 
import os

''' to combine regional models into one meta model and project forward
    - simulate travels
    - vaccine scenario 
    - with chosen feature (use job_id) '''

features=['prevalence_per_million','cases','case_in_next_3months','prevalence_in_next_3months']

job_id = 0 

feature=features[job_id]
print('feature',feature)
fits=pd.read_csv("curve_fitting_results_83_before_vaccine.csv")
#data=pd.read_csv("prevalance_84_cumcase.csv")

# some regions need extra tuning for their S2A
s2a_adjust=pd.read_csv("baseline_projection_S2A_adjust.csv")

fits.loc[197,'location']="China"
fits.loc[197,'start']=298
fits.loc[197,'offset']=63.8747316757306

# read in regional models 
abm=[]
for i in range(197):
    r=fits.location.to_numpy()[i]
    file= '/home-4/yhuang98@jhu.edu/scratch/travel_single_region_after_vac_senior_first_83/abm_'+str(i)+'_bounds_.pkl'    
    if not os.path.exists(file):
        file='/home-4/yhuang98@jhu.edu/scratch/travel_single_region_after_vac_senior_first_83/abm_'+str(i)+'.pkl'
    else:
        print(r,'new fit')
    with open(file,'rb') as f:
        t=pickle.load(f)
        
    abm.append(t)

# China doesn't come from fitting 
with open('/home-4/yhuang98@jhu.edu/scratch/abm_china.pkl','rb') as f:
    t=pickle.load(f)
abm.append(t)   

seed=0
np.random.seed(int(seed))

print('number of regions',len(abm))
# some initial adjustment
for i in range(len(abm)): 
    r=abm[i].region
     # if S2A needs to be adjusted: 
    if r in s2a_adjust.region.to_numpy():
        s=s2a_adjust.loc[s2a_adjust.region==r,'scale'].item()
        abm[i].params['S2A']=abm[i].params['S2A']*s
        print(r,' adjust s2a ',s)
        
    # not simulate travel for individual region (but for all regions together)
    abm[i].sim_travel=False
    
# init meta model     
meta=Meta_ABM(ABM=abm)

# read in vaccine scenario
vaccine=pd.read_csv("vaccine_plan_priority_"+feature+"_83.csv")

print('covax aid:')
# update vaccine dosage for all regions
for i in range(len(abm)):
    r=meta.ABM[i].region
    row=vaccine.loc[vaccine.location==r]
    
    if row.shape[0]==0: continue 
    vac=row['tot_vac']
    vac=int(np.round(vac/2)/meta.ABM[i].params['actual_pop_scale'])
    print(r,'old vac:',meta.ABM[i].vaccine_plan['Pfizer'],'; new vac:',vac)
    meta.ABM[i].vaccine_plan={'total':2*vac,'Pfizer':vac,'J&J':vac}
    meta.ABM[i].vaccine_date=0
    print(r,meta.ABM[i].vaccine_plan)
    print()
    
# run meta model
meta.run(steps=92)
    
# collect data     
def data_from_model(meta):
    res=[]
    n=len(meta.ABM)
    for i in range(n):
        result={}
        result['region']=meta.ABM[i].region
        result['actual_pop_scale']=meta.ABM[i].params['actual_pop_scale']
        result['data']=meta.ABM[i].data
        res.append(result)
    return res

meta=data_from_model(meta) 

with open('vaccine_scenarios_83/meta_priority_'+feature+'_data_'+str(int(seed))+'.pkl','wb') as f:
    pickle.dump(meta,f)