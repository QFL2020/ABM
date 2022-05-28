from ABMpy_region.model import Meta_ABM 
from ABMpy_region.agent import ABM
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict 
import os
import sys

''' continue to evolve regional models independently
    - update regional demographics 
    - add vaccination
    - add effect of travels, from data 
    - adjust S2A and find best-fit 
    - evolve until 7/1/2021 '''


# helper functions
def make_data(start,end):
    # get the case data between start and end dates
    data_r=[]
    row=data[data.location==r]
    for i in range(int(start)+1,int(end)+1):

        data_r.append(row[str(int(i))].item())
    data_r=np.array(data_r)*10**6
    return data_r

def rmse(a1,a2):    
    return ((a1[-5:]-a2[-5:])**2).sum()


job_id = 0


# input needed: bestfit results, vaccine info, data
fits=pd.read_csv("curve_fitting_results_83_before_vaccine.csv")
vaccine=pd.read_csv("vaccine_info_83.csv")
data=pd.read_csv("prevalance_712_cumcase.csv")
 
i=job_id
r=fits.location.to_numpy()[i]
print(r,job_id)


# if output exist, exit
file='/home-4/yhuang98@jhu.edu/scratch/travel_single_region_after_vac_senior_first_83/abm_'+str(job_id)+'_plots.pkl'
if os.path.exists(file) :
    print('done')
    sys.exit()
    
# read in model
with open('/home-4/yhuang98@jhu.edu/scratch/travel_single_region_before_vac_83/abm_'+str(i)+'.pkl','rb') as f:
    m=pickle.load(f)
    
with open("ABM_country_input_all.pkl",'rb') as f:
    demo=pickle.load(f)

# demo info    
demo_dist=demo[r]['DEMO_DIST']
scale_s2a=demo[r]['SCALE_FACTOR_S2A']
case_fatality=demo[r]['CASE_FATALITY']
case_frac=demo[r]['CASE_FRAC']
# reset demo
m.demo_reset(demo_dist=demo_dist,case_fatality=case_fatality,case_frac=case_frac)

# dates info
current_day=fits[fits.location==r].end.item() #current global date in simulation
fit_to=546 # 7/1/2021 # days since 1/1/2020
sim_start_day=m.data.shape[0] #current day in sim
sim_end_day=fit_to-current_day+sim_start_day # end day in sim

offset=fits[fits.location==r].offset.item()

# vaccine info
row=vaccine.location==r
change_vac=False

# vaccinate senior first 
m.vaccine_senior_first=True

# set up vaccine plan depending on region's vaccination coverage

# 1) have reached 3% by 5/31/2021
    # vacccine dosage between 3% and 5/31
if (not vaccine.loc[row,'dailyVac1_permillion_bw3andLastMonth'].isnull().item()):
    print('1st vac')  
    change_vac=True
    change_vac_day = 516-current_day
    vac=int(vaccine.loc[row,'dailyVac1_permillion_bw3andLastMonth'].item())
    
    if vac>0:
        m.vaccine_plan={'total':2*vac,'Pfizer':vac,'J&J':vac}
        # vaccinate 3% population on a single day
        m.vaccine_coverage(frac1=0.015,frac2=0.015)
        m.vaccine_date=m.data.shape[0]-1
    else:
        m.vaccine_date=m.data.shape[0]+change_vac_day
    # vacccine dosage between and 5/31 and 6/30
    vac2=int(vaccine.loc[row,'dailyVac1_permillion_duringLastMonth'].item())
    
# 2) not reached 3% but started vaccination    
elif (not vaccine.loc[row,'dailyVac1_permillion_duringLastMonth'].isnull().item()):
    print('2nd vac')
    # vacccine dosage between and 5/31 and 6/30
    vac=int(vaccine.loc[row,'dailyVac1_permillion_duringLastMonth'].item())
    m.vaccine_plan={'total':2*vac,'Pfizer':vac,'J&J':vac}
    frac1=vaccine.loc[row,'vac1RateCumu_asOfLastMonth'].item()/100.
    m.vaccine_coverage(frac1=frac1,frac2=frac1)
    m.vaccine_date=m.data.shape[0]-1
    
# 3) not vaccination yet   
else: pass

# S2A scale factor range
test=np.linspace(0.1,1.8,35)
#test=np.linspace(0.1,1,10)
#test=np.hstack((test,np.array([1.1])))
#test=np.hstack((test,np.linspace(1.2,4,29)))

# set S2A
s2a=m.params['S2A']
s2a=s2a*test

# update travel
m.sim_travel=True
# prepare prevalence dataframe in the ABM object 
m.travel_prep()
m.params['c_I']=800#*10
m.params['c_D']=1000


# reset travel scale
new_cases_daily=(data.loc[data.location==r,'516']-data.loc[data.location==r,'496']).item()*10**6/20
scale=min(1,new_cases_daily/40) # only 1/40 cases come from importation 
print('travel scale',scale)
print('base S2A',m.params['S2A'])
print('offset',offset)
m.update_travel_scale(scale)

# prepare data
data_r=make_data(current_day,fit_to)

m_sims=[]
min_rmse=[]

print('test',test)

# prepare data for plotting
plot_start_day=max(current_day-50,0)
if plot_start_day-fits[fits.location==r].start.item()<0:
    plot_start_day=fits[fits.location==r].start.item()
days=np.arange(plot_start_day+1,fit_to+1)

data_r_plot=make_data(plot_start_day,fit_to)

plots={}
plots['data_r']=data_r_plot 
plots['days']=days
plots['sims']=[]

# find exact days to run sim
days=days-int(fits[fits.location==r].start.item())

# loop over all S2A and find the one with min rmse
for s in s2a:
    np.random.seed(0)
    m_sim=copy.deepcopy(m)
    m_sim.params['S2A']=s
    if change_vac: # change dosage during runs, for regins that have reach 3% coverage
        m_sim.run(steps=int(change_vac_day))
        print('change vac')
        m_sim.vaccine_plan={'total':2*vac2,'Pfizer':vac2,'J&J':vac2}
        m_sim.run(steps=int(sim_end_day-sim_start_day-change_vac_day))
    else:
        m_sim.run(steps=int(sim_end_day-sim_start_day))
    m_sims.append(m_sim)
    #break #
    sim_r=m_sim.data.loc[sim_start_day:sim_end_day,'cum_case'].to_numpy()+offset
    plots['sims'].append(m_sim.data.loc[days,'cum_case'].to_numpy()+offset)
   

    rmse_= rmse(data_r,sim_r)
                                                         
    min_rmse.append(rmse_)
    min_i=min_rmse.index(min(min_rmse))
    for j in range(len(min_rmse)):
        if j!=min_i:
            m_sims[j]=0
    if len(min_rmse)> 25 and min_i<len(min_rmse)-24:
        break

        
# save results 
min_i=min_rmse.index(min(min_rmse))
plots['best_sim']=min_i
plots['best_test']=test[min_i]
print('min sim number',min_i,'of',len(min_rmse),';','rmse',min(min_rmse))

with open('/home-4/yhuang98@jhu.edu/scratch/travel_single_region_after_vac_senior_first_83/abm_'+str(job_id)+'.pkl','wb') as f:
    pickle.dump(m_sims[min_i],f)


with open('/home-4/yhuang98@jhu.edu/scratch/travel_single_region_after_vac_senior_first_83/abm_'+str(job_id)+'_plots.pkl','wb') as f:
    pickle.dump(plots,f)



