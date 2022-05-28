import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import pickle
import time
from functools import lru_cache
#from numba import njit
#from linetimer import CodeTimer
from ABMpy_region.util import smap
from ABMpy_region.agent import ABM
from multiprocessing import Pool
import os
from ABMpy_region.input_travel import *
from collections import defaultdict 

class Meta_ABM:
    ''' global model
        consists of a list of individual ABM, handles their travels '''

    def __init__(self,ABM=[]  ):
        
        self.ABM=ABM  # list of ABM models for different regions
     
        ### shut off single region travel sim
        
        self.global_pop_scale=0
        for i in range(len(ABM)):
            self.ABM[i].sim_travel=False
            self.global_pop_scale+=self.ABM[i].params['actual_pop_scale']
            
        ''' data collection'''
        self.compartments=np.arange(5)
        self.data_columns=[i for i in range(5)]
        self.data_columns.extend(['new_cases','cum_case','slence','vaccinated'])
        self.compartment_names=['S','A','I','R','D']              
        self.data = pd.DataFrame(columns=self.data_columns)
        self.update_compartments()
        
        
        # mapping of {region: region_ID}
        self.region_ID_sim={self.ABM[i].region:i for i in range(len(ABM))}
    

            
    def step(self,multiprocessing=False):
        '''
        daily evolution
        1) travel
        2) run regional models
        3) update dataframe
        '''
        
        
        ''' travel '''
        #self.travel_replace()
        self.travel_simple()
        
        ''' model evolve by region'''
        regions=range(len(self.ABM))
        
        if multiprocessing:
            num_process= min(os.cpu_count(),len(regions))
            print('num_process',num_process)
            pool=Pool(processes=num_process)
            res=pool.map(smap, [self.ABM[i].step for i in regions])   
            #print(res)
            self.ABM=res
        else:
            for i in regions:
                print('region',i)
                self.ABM[i].step()
                
        '''update global counts'''
        self.update_compartments()
        
    def run(self,steps=1,multiprocessing=False):
        ''' run model for steps'''
        for i in range(steps):
            self.step(multiprocessing=multiprocessing)
    
    def update_compartments(self):
        '''keeps record of global counts'''
        nregions=len(self.ABM)
        regions=range(nregions)
        df_current=pd.DataFrame(columns=self.data_columns)
        
        for i in regions:
            df_current=df_current.append(self.ABM[i].data.iloc[-1:]*self.ABM[i].params['actual_pop_scale'], ignore_index=True)
        df_current['prevalence']=df_current['prevalence']/self.global_pop_scale
        self.data=self.data.append(df_current.sum(),ignore_index=True)
        
        
    def plot_compartments(self):
        
        ''' plot compartment time series'''
        fig, ax = plt.subplots()
        t=np.arange(self.data.shape[0])
        
        color=['green','gold','chocolate','cyan','red']
        color_dict={c:co for c,co in zip(self.compartments,color)}
        
        for c in self.compartments[1:]:
            ax.plot(t,self.data[c],color=color_dict[c],label=self.compartment_names[c])
        ax.legend(loc='best',frameon=False)
        ax.set_xlabel('step')
        ax.set_ylabel('population')
        plt.show()
        
        
    def plot_data(self,col=['new_cases']):
        ''' plot just one column from data frame'''
        for c in col:
            plt.plot(self.data[c],label=c)
            plt.xlabel('step')
            plt.ylabel('population')
            
        plt.legend()
        plt.show()
        

    def travel_simple(self):
        ''' travel
         importations from neighbors based on simulated prevalence and actual population scale
        - calculate coutry level prevelance
        - scale to actual population size
         '''
        
        # dictionary for country prevalence
        p_region={'US':0}
        risk_I={} # international
        risk_D={} # domestic
        
        
        # get prevalence for each country
        for r in self.region_ID_sim:     
            pop_scale=self.ABM[self.region_ID_sim[r]].params['actual_pop_scale']
            
            p_r=self.ABM[self.region_ID_sim[r]].data.iloc[-1]['new_cases']/10**6 * pop_scale
            p_region[r]= p_r
            
            # if region is a US state, add it to US prevalence
            if r.startswith('US'):
                p_region['US']+=p_r
        print(p_region)  
        
        # risk_I per country, based on neighbor prevalence
        # risk_D per US state
        for r in self.region_ID_sim: 
            risk_I[r]=0
            risk_D[r]=0
            if r.startswith('US.'):
                for neighbor_state in weight_D[r]:
                    if neighbor_state in self.region_ID_sim:
                        risk_D[r]+=weight_D[r][neighbor_state]*p_region[neighbor_state]
                for neighbor_country in weight_I['US']:
                    if neighbor_country in self.region_ID_sim:
                        risk_I[r]+=weight_I['US'][neighbor_country]*p_region[neighbor_country]  
            else:        
                for neighbor_country in weight_I[r]:
                    if neighbor_country in self.region_ID_sim:
                        risk_I[r]+=weight_I[r][neighbor_country]*p_region[neighbor_country]  
                        
        print('international',risk_I)
        print('demo',risk_D)
        
        # total risk per region 
        for r in self.region_ID_sim:
            abm=self.ABM[self.region_ID_sim[r]]
            risk=abm.params['c_I']*risk_I[r]+abm.params['c_D']*risk_D[r]
            risk=int(np.round(risk))
            risk=min(risk,6)
            print(r,risk)
            # importation
            abm.travel_importation(num=risk)
                
        
        
                
    def travel_replace(self):
        ''' 
            NOT USED
            simulate travels from j->i 
            by replacing attributes of agents in i by attributes of agents in j, 
            without actually moving agents from j to i
            conserving number of agents in all regions
        '''
        regions=range(len(self.ABM))
        
        # record eligible agents in each region
        agents_in_i=[]
        for i in  regions: 
            # anyone that's in i and not dead
            cond=self.ABM[i].agents['state']!=4 #==0
            agents_in_i.append(np.where(cond)[0])
           
        for i in regions:
            
            # number of importations to i
            num_travel=self.travel_matrix[i,i]
            travellers=np.random.choice(agents_in_i[i],size=num_travel,replace=False)
            np.vectorize(self.ABM[i].update_region_group_mixing)(agent=travellers)
   
            m=0 # dummy var to keep track of travellers
            for j in regions: # orgins
               
                if j!=i:
                    num_j_to_i = self.travel_matrix[j,i]
                    agents_j_to_i = travellers[m:m+num_j_to_i]
                    
                    # randomly choose agents in j to be copied to i
                    copies_from_j= np.random.choice(agents_in_i[j],size=num_j_to_i,replace=False)
                    
                    # replacing attributes of copies_from_j to agents_j_to_i
                    for a in range(num_j_to_i):
                        self.agent_replace(region1=i,region2=j, \
                                           agent1=agents_j_to_i[a],agent2=copies_from_j[a])
                    
                    m+=num_j_to_i       
        