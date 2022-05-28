from ABMpy_region.model import Meta_ABM 
from ABMpy_region.agent import ABM
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 

''' with best-fit params, evolve regional models to desired end date
    - without vaccination
    - without travels '''

# read in fit results
fit=pd.read_csv("curve_fitting_results_83_before_vaccine.csv")
 
job_id = 0
i=job_id
r=fit.location.to_numpy()[i]

# default params, same for all regions for now
sim_pop=10**6
params={
        'A0':122,'I0':100,
        'A2I':0.45,
        'max_mix':30,
        'region':0,
        'country':0,
        'c_I':100,
        'c_D':100,
        'actual_pop_scale':1,
        'sim_travel':False,
        'global_day':200
        
        }
params['nPop']=sim_pop


# when to start vaccination (same for all regions for now) 
# date > end date, so no vaccine in sims yet 
params['vaccine_date']=1000 
# how many vaccines per day, just placeholder
params['vaccine_plan']={'total':20,'Pfizer':10,'J&J':10}


# read in bestfit params
fit_params=['group_size', 'gamma', 'gamma_M', 'S2A','global_day','actual_pop_scale']
if r.startswith("US."):
    params['country']='US'
else:
    params['country']=r
    
params['region']=r
for p in fit_params:
    params[p]=fit[fit.location==r][p].values[0]
params['group_size']=int(params['group_size'])
params['global_day']=int(params['global_day'])


print('job_id',job_id)
print('day since 1/1/2020:',params['global_day'])
# divide by 10**3 to get the scale factor as if model is set to be 1M

# convert "min_sim_number" from grid search to random seed
seed=fit[fit.location==r].min_sim_number.item()%10
np.random.seed(int(seed))

# run model
m=ABM(params=params)

days=int(fit.loc[fit.location==r,'end'].item()-fit.loc[fit.location==r,'start'].item())   
m.run(steps=days)

# save results
with open('/home-4/yhuang98@jhu.edu/scratch/travel_single_region_before_vac_83/abm_'+str(i)+'.pkl','wb') as f:
    pickle.dump(m,f)