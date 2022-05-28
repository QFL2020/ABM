import numpy as np
import pandas as pd
from ABMpy_region.input import *
import time
from functools import lru_cache
from collections import defaultdict 
#from linetimer import CodeTimerjh
from ABMpy_region.util import init_array, random_choice,choice
from ABMpy_region.input_travel import *
import matplotlib.pyplot as plt


class ABM():
    
    ''' regional model '''
    
    def __init__(self,params = None ):

        # input params
        self.params=params # param dictionary
        
        ''' region '''
        self.country=params['country']
        self.region=params['region']
        print(self.region)
        self.nPop=int(params['nPop'])
        
        ''' travel '''
        self.global_day=params['global_day'] # global day counter from 1/1/2020
        self.sim_travel=params['sim_travel'] #boolean 
        if self.sim_travel:
            # prepare prevalence data for importation
            self.travel_prep()
        
        if 'travel_scale' in self.params.keys():
            self.travel_scale=self.params['travel_scale']
        else: self.travel_scale=1
        
        ''' epidemiology '''
        self.gamma=params['gamma'] # param for scaling down num_mix of agents from other groups
        self.group_size=params['group_size']
        self.gamma_m=params['gamma_M'] # param for scaling down num_mix for agents in I
        self.A2I=params['A2I']
        A0=params['A0']
        I0=params['I0']
        max_mix=params['max_mix']
     
        ''' initialize agents and their attributs
         in the form of a dictionary of numpy arrays '''
        agents={}  
        agents['ID']=np.arange(self.nPop)
        
        agents['state']=init_array(0,self.nPop) # set all states to be "S"
        agents['timeA']=init_array(0,self.nPop)
        agents['timeI']=init_array(0,self.nPop)
        
        # num_mix
        agents['num_mix']= np.random.choice(np.arange(1,max_mix+1),size=self.nPop,replace=True)
        
        # eventual state after A or I
        agents['state_after_A']=init_array(-1,self.nPop)
        agents['time_in_A']=init_array(0,self.nPop)
        agents['state_after_I']=init_array(-1,self.nPop)
        agents['time_in_I']=init_array(0,self.nPop)
        
        agents['infectious_yn']= init_array(0,self.nPop)
        agents['timeA_infectious']=init_array(100,self.nPop)
        agents['infectious_period']=init_array(0,self.nPop)
        agents['second_infection']=init_array(0,self.nPop)
        agents['generation']=init_array(0,self.nPop) # the generation number of infected agents, start from 1
        agents['I0']=init_array(0,self.nPop)
        # init current_region
        agents['region'] = init_array(params['region'],self.nPop)
        
        ''' demographic, for individuals and regional distribution'''    
        sex=range(2) # ['F','M']
        age=range(3) #['Child','Adult','Senior']
        race=range(2) # ['reference','minority']
        
        # demo=12 combinations of [sex,age,race], shape = 12x3
        demo=np.array(np.meshgrid(sex,age,race)).T.reshape(-1,3)
        
        # demo_group number
        demo_group=np.arange(12) # 12 demographic groups in total
        
        # demographic distribution
        if 'DEMO_DIST' in params.keys():
            self.DEMO_DIST=params['DEMO_DIST']
        else:
            self.DEMO_DIST=DEMO_DIST
        self.DEMO_DIST=self.DEMO_DIST/self.DEMO_DIST.sum()
        
        # agent demographic assignment
        agent_demo=np.random.choice(demo_group,size=self.nPop,p=self.DEMO_DIST)
        agents['demo_group']=agent_demo
        agent_demo=demo[agent_demo]
        agents['sex']=agent_demo[:,0]
        agents['age']=agent_demo[:,1]
        agents['race_higher_risk']=agent_demo[:,2]
        
        # demographic dependence of disease
        if 'SCALE_FACTOR_S2A_AGE' in params.keys():
            self.SCALE_FACTOR_S2A = params['SCALE_FACTOR_S2A']
        else:
            self.SCALE_FACTOR_S2A=SCALE_FACTOR_S2A 
        
        if 'CASE_FATALITY' in params.keys():
            self.CASE_FATALITY = params['CASE_FATALITY']
        else:
            self.CASE_FATALITY=CASE_FATALITY
           
        # health
        #agents['health_yn'] = np.random.choice(['Y','N'],size=self.nPop,p=HEALTHY_DIST)
        

        ''' vaccine params '''
        agents['vaccinated']=init_array(0,self.nPop) # agent vaccination status
        agents['risk_reduction']=init_array(1.,self.nPop) # (1-vaccine efficacy)
        
        # vaccine plan 
        self.vaccine_date=params['vaccine_date']
        self.vaccine_plan=params['vaccine_plan']
        self.vaccine_reduction=VACCINE_RISK_REDUCTION #{'Pfizer':0.05,'J&J':0.342}
        
        # indicator of whether all agents in a region have been vaccinated
        self.vaccine_all=False
        # indicator for whether to prioritize seniors
        self.vaccine_senior_first =False
        
        
        ''' grouping for agent meetings '''
        
        # init groups of agents, init value = 0
        agents['group']=init_array(0,self.nPop)
        self.groups=[]
        
        # calculate number of groups
        num_group=self.nPop//self.group_size    
        # if group_size > pop_r, everyone is in group 0
        if num_group ==0:
            self.groups.append(0)
              
        else:
            # the remainder is the size of the last group
            extra=self.nPop%self.group_size
            
            # set up the array of groups
            whole_groups=np.arange(num_group)
            self.groups.extend(whole_groups)
            whole_groups=np.repeat(whole_groups,self.group_size)
            
            # add the last group if extra>0
            if extra>0:
                self.groups.append(whole_groups[-1]+1)
                last_group=np.tile(whole_groups[-1],extra)
                groups = np.append(whole_groups,last_group)
            else:
                groups=whole_groups
            
            # map agents to group assignment
            agents['group']=groups

#######################    
        ''' initiate infected agents '''
        
        # agents to be infected
        init_agents= np.random.choice(agents['ID'],size=A0+I0,replace=False)
        # agents who are in A
        agents['state'][init_agents[:A0]]=1
        agents['timeA'][init_agents[:A0]]=np.random.binomial(5,0.5,size=A0)
        # are they currently infectious?
        for i in init_agents[:A0]:
            if agents['timeA'][i]>=2:  
                agents['infectious_yn'][i] =1
        
        # agents who are in I
        agents['state'][init_agents[A0:A0+I0]] = 2
        agents['infectious_yn'][init_agents[A0:A0+I0]]=1
        agents['timeI'][init_agents[A0:A0+I0]]=np.random.binomial(8,0.5,size=I0)
        agents['num_mix'][init_agents[A0:A0+I0]]=np.round(max_mix/2)*self.gamma_m
        agents['I0'][init_agents[A0:A0+I0]]=1
        agents['generation'][init_agents]=1 # first generation of I agents
        
        
        self.agents=agents
        # the next states of infected agents 
        if A0>0:
            np.vectorize(self.A2I_or_R)(agent=[init_agents[:A0]])
        if I0>0:
            np.vectorize(self.I2R_or_D)(agent=[init_agents[A0:A0+I0]])

#######################                
        ''' init mixing array for meetings'''
        self.mixing_array_region=np.array([])
        
        # mixing array by region-group for same-group mixing
        self.mixing_array_region_group=defaultdict()
        
        # init need_update_region_group
        self.region_group_need_update_mixing={g:True for g in self.groups}
     
    
        ''' dataframe for data collection'''
        self.compartments=np.arange(5)
        self.data_columns=[i for i in range(5)] # columns of 5 states
        self.data_columns.extend(['new_cases','cum_case','prevalence','vaccinated'])
        self.compartment_names=['S','A','I','R','D']              
        self.data = pd.DataFrame(columns=self.data_columns)
        # update dataframe
        self.update_compartments(new_cases=I0)

        
    def demo_reset(self,demo_dist=None,case_fatality=None,case_frac=None):
        '''
            input: numpy arrays of length 12 to specify the value of each demographic group
            
            demo_dist: demographic proportion of each demo group
            case_fatality: likewise
            case_frac: the proportion of each demo group in all cases
            
            this is a function for 
            - reseting demographic distribution
            - adjusting S2A such that the overall S2A of the region stays the same
            - resetting agents' demographic attributes to be consistent with new demo_dist, case_fatality and case_frac
            
        '''
        
        # normalize the distribution if it is not yet normalized
        self.DEMO_DIST=demo_dist
        self.DEMO_DIST=self.DEMO_DIST/self.DEMO_DIST.sum()
     
        old_s2a=self.params['S2A'] 
        scale_s2a=case_frac/demo_dist # a relative susceptibility to be infected for each demo group
                                       # given the new case_frac and demo_dist
        
        # handeling null values
        for i in range(12):
            if np.isnan(scale_s2a[i]):
                scale_s2a[i]=0
            
        # rescale scale_s2a such that scale_s2a[adult female senior]=1
        scale_s2a=scale_s2a/scale_s2a[2]
        self.SCALE_FACTOR_S2A=scale_s2a   
        
        # compute new base S2A (that of adult female senior)
        # put it in the params dictionary
        self.params['S2A']=old_s2a*(DEMO_DIST*SCALE_FACTOR_S2A).sum()/  \
                              (self.DEMO_DIST*self.SCALE_FACTOR_S2A).sum()
        # update case fatality
        self.CASE_FATALITY=case_fatality
        
        ''' redistribute demographic attributes to agents, 
            in order to maintain consistency to the new demo_dist, case_fatality and case_frac '''
        
        # current agents in S
        s=self.agents['state']==0 
        healthy_agents=np.where(s)[0]
        
        d=self.agents['state']==4
        dead_agents=np.where(d)[0] # current dead agents
        sick_agents=np.where((~s) & (~d))[0] # agents that have been sick 
        
        case_proportion=len(sick_agents)/self.nPop # acumulative case proportion
        death_proportion_of_sick=d.sum()/(~s).sum() 
        
        case_frac=case_frac/case_frac.sum() # normalize case_frac if it is not yet normalized
        
        # proportion of each demo group in agents that have not been sick
        healthy_dist=self.DEMO_DIST-case_proportion*case_frac
        healthy_dist=healthy_dist/healthy_dist.sum()
        
        # proportion of each demo group in dead agents 
        death_dist=case_frac*case_fatality
        death_dist=death_dist/death_dist.sum()
        
        # proportion of each demo group in agents that have been sick but not dead 
        case_not_death_frac=case_frac-death_proportion_of_sick*death_dist
        case_not_death_frac=case_not_death_frac/case_not_death_frac.sum()
        
        
        ''' reset demo properties to agents'''        
        sex=range(2) # ['F','M']
        age=range(3) #['Child','Adult','Senior']
        race=range(2) # ['reference','minority']
        demo=np.array(np.meshgrid(sex,age,race)).T.reshape(-1,3)
        demo_group=np.arange(12)
        
        np.random.seed(0)
        
        def set_dist(ids,dist):  
            ''' helper function for setting demo attributes in agents in ID list "ids", 
                according to distribution "dist" '''
            groups=np.random.choice(demo_group,size=len(ids),p=dist)
            #print(groups)
            self.agents['sex'][ids]=demo[groups][:,0]
            self.agents['age'][ids]=demo[groups][:,1]
            self.agents['race_higher_risk'][ids]=demo[groups][:,2]
            self.agents['demo_group'][ids]=groups
            
        set_dist(healthy_agents,healthy_dist)  
        set_dist(sick_agents,case_not_death_frac)
        set_dist(dead_agents,death_dist)
        
        return
    
    def update_travel_scale(self,scale):
        ''' helper to update travel scale'''
        self.travel_scale=scale
        
    def update_compartments(self,new_cases=None):
        '''helper to update the compartment dataframe'''
        a=self.agents
        counts={}
        for c in self.compartments:
            # count agents in compartment c 
            state_c = a['state']==c
            counts[c]= [(state_c).sum()]

        counts['new_cases']= new_cases
        counts['vaccinated']= (a['vaccinated']==1).sum()
        if  self.data.shape[0]>0:
            cum_case=self.data.cum_case.iloc[-1]+new_cases
        else:
            cum_case=new_cases
        counts['cum_case']=cum_case
        
        # compute prevalence
        cond=self.agents['infectious_yn']==1
        num_I=cond.sum()
        counts['prevalence']=num_I/self.nPop
        df=pd.DataFrame(data=counts)
        self.data=self.data.append(df, ignore_index=True)
        
        

    def make_mixing_array_np(self):
        
        '''
            generate mixing array (to speed up meetings) at the region level and at the group
            for group level mixing array, only update when 
                region_group_need_update_mixing[region][group] = True
        '''

        ID_r = self.agents['ID']
        num_mix_r = self.agents['num_mix']
        group_r=self.agents['group']
        
        # always update the mixing array at the region level
        self.mixing_array_region= np.repeat(ID_r,num_mix_r)

        #only update group level if need_update_group=True
        for group in self.groups:
            if self.region_group_need_update_mixing[group]:
                same_group =  group_r == group 
                ingroup = np.where(same_group)
                ID_g=ID_r[ingroup]
                num_mix_g=num_mix_r[ingroup]
                if len(ID_g)>2:
                    self.mixing_array_region_group[group]= np.repeat(ID_g,num_mix_g)
                else:
                    self.mixing_array_region_group[group]=ID_g

                self.region_group_need_update_mixing[group]=False

                                          
    def update_region_group_mixing(self,agent=0):
        '''
            set region_group_need_update_mixing[region][group]=True
            so mixing array for the group the input agent is in will get updated in next step
        '''
        
        group=self.agents['group'][agent]
    
        self.region_group_need_update_mixing[group]=True
             
    def A2I_or_R(self,agent=0):
        '''
            decide the outcome of agents in A
        '''

        A2I=self.params['A2I']
                                                
        if np.random.rand()<A2I:
            
            self.agents['state_after_A'][agent]=2
            self.agents['time_in_A'][agent]=min(np.round(np.random.lognormal(mean=1.63,sigma=0.5)),21)
            self.agents['timeA_infectious'][agent]= max(self.agents['time_in_A'][agent]-2,0)
        else: 
            self.agents['state_after_A'][agent]=3
            self.agents['time_in_A'][agent]=min(np.round(np.random.lognormal(mean=2.23,sigma=0.2)),21) #mean 9.5
            self.agents['timeA_infectious'][agent]= 3
    
    def I2R_or_D(self,agent=0):
        '''
            decide the outcome of agents in I
        '''
        
        #I2D=CASE_FATALITY[self.agents['sex'][agent]][self.agents['age'][agent]]
        I2D=self.CASE_FATALITY[self.agents['demo_group'][agent]]
        if np.random.rand()<I2D:
            self.agents['state_after_I'][agent]=4
            
            #self.agents['time_in_I'][agent]=min(np.round(np.random.lognormal(mean=1.95,sigma=0.4)),21) #median 7
            
        else:
            self.agents['state_after_I'][agent]=3
      
        self.agents['time_in_I'][agent]=min(np.round(np.random.lognormal(mean=2.04,sigma=0.4)),21)#mean 8
         
        
    def update_state(self,agent=0,new=1):
        '''update state of an agent '''
        
        #print('agent update',agent,new)
        old_state = self.agents['state'][agent]
        self.agents['state'][agent] = new
        #print(self.agents['state'][agent])
        
        # state A
        if new == 1:
            # decide the next state after A
            self.A2I_or_R(agent=agent)
        
        # state I
        if new == 2:
            # agent must be infectious
            self.agents['infectious_yn'][agent]=1
            # num_mix reduces by gamma_m
            self.agents['num_mix'][agent] = np.round(self.agents['num_mix'][agent]*self.gamma_m)
            # decide the next state after I
            self.I2R_or_D(agent=agent)
            self.update_region_group_mixing(agent=agent)
        
        # state R or D
        if new >2:
            # no longer infectious
            self.agents['infectious_yn'][agent]=0
            if old_state==2: # if R
                # reset num_mix
                self.agents['num_mix'][agent] = np.round(self.agents['num_mix'][agent]/self.gamma_m)
                self.update_region_group_mixing(agent=agent)
        
        #if new == 4:
            #self.update_region_group_mixing(agent=agent)
         #   self.agents['infectious_yn'][agent]=0
                   
    def one_v_one_meeting(self,agent1=0,agent2=1):
        '''
        meeting between agent 1&2
        1 is infectious 
        '''
        s2= self.agents['state'][agent2]
        #print('meeting',agent1,agent2)
        
        # return if s2 is not S
        if s2>0:
            return
        
        # get S2A for the demo group of 2
        demo=self.agents['demo_group'][agent2]
        S2A=self.params['S2A']
        S2A=S2A*self.SCALE_FACTOR_S2A[demo]*self.agents['risk_reduction'][agent2]
        
        prob = np.random.rand()
        
        if prob < S2A:
            # 2 is infected
            self.update_state(agent=agent2,new=1)
            
            # keep some records
            self.agents['second_infection'][agent1]+=1
            self.agents['generation'][agent2]=self.agents['generation'][agent1]+1
             
 
        return
    

    def one_v_many_meetings_on(self,agent):
        '''
            agent meents other agents randomly O(n) verion
            
        '''
        num_mix=self.agents['num_mix'][agent]#*2
        group=self.agents['group'][agent]
        
        # number of agents to meet from the same group
        ingroup_num_mix=np.random.binomial(num_mix,self.gamma)
        
        # number of agents to meet regardless of group
        outgroup_num_mix=num_mix-ingroup_num_mix
        
        # same group meetings:
        n=len(self.mixing_array_region_group[group])
        
        for k in np.arange(ingroup_num_mix):
            # randomly choose from mixing array
            j=choice(self.mixing_array_region_group[group],n)
            self.one_v_one_meeting(agent1=agent,agent2=j)
                 
        
        # out of group meetings:
        n=len(self.mixing_array_region)
        for k in np.arange(outgroup_num_mix):
            j=choice(self.mixing_array_region,n)
            z=0
            while j==agent and z <5:
                j=choice(self.mixing_array_region,n)
                z+=1
            self.one_v_one_meeting(agent1=agent,agent2=j)
             
    
    def meeting(self):
        ''' loop over all infectious agents to initiate meetings '''
        t1=time.time()
    
        # create/update mixing array
        self.make_mixing_array_np()
        
        print('update mixing array',np.round((time.time()-t1),4),'s')
        
        # Infectious agents meet others 
        cond=self.agents['infectious_yn']==1
        num_I=cond.sum()
        I_group = np.where(cond)[0]
        
        t0=time.time()
       
        for a in I_group:
            self.one_v_many_meetings_on(a)

        print('one v many meeting',np.round((time.time()-t0),4),'s, average',
              np.round((time.time()-t0)*1000/max(len(I_group),1),4),'ms')
      


    def step(self):
        '''
        daily evolution
        1) meeting
        2) vaccination
        3) update agents by state
        4) update dataframe
        '''
        if self.sim_travel:
            self.travel_simple()
        
        print('step ', self.data.shape[0])
        self.meeting()
        
        # if after vaccine start date
        if self.data.shape[0]>=self.vaccine_date:# and (not self.vaccine_all):
            print('vac')
            self.vaccine()
        new_cases=self.agents_evolve()
        self.update_compartments(new_cases=new_cases)  
        # update counter
        self.global_day+=1
        
        return self
        
    def agents_evolve(self):
        '''
            evolve all agents one step forward
        '''
        
        ''' agents in A'''
        stateA = self.agents['state']==1
        stateA_index=np.where(stateA)[0]
        #new_A = (stateA & (self.agents['timeA']==0)).sum()
        
    
        # timeA+1
        self.agents['timeA'][stateA_index]+=1
   
        agent_infectious= self.agents['infectious_yn']==1
        self.agents['infectious_period'][agent_infectious]+=1
        # update infectiousness
        stateA_infectious= stateA & (self.agents['timeA']>=self.agents['timeA_infectious'])#10
        stateA_infectious_index=np.where(stateA_infectious)[0]
        self.agents['infectious_yn'][stateA_infectious_index]=1
        if len(stateA_infectious_index)>0:
            np.vectorize(self.update_region_group_mixing)(agent=stateA_infectious_index)
        
        # if time_in_A is reached, move on to the next state
        stateA_evolve= stateA & (self.agents['timeA']>=self.agents['time_in_A'])
       
        stateA_evolve_index = np.where(stateA_evolve)[0]
        if len(stateA_evolve_index)>0:
            #print('stateA_evolve_index',stateA_evolve_index)
            np.vectorize(self.update_state)(
                agent=stateA_evolve_index,
                new=self.agents['state_after_A'][stateA_evolve_index])
       
    
        ''' agents in I '''
        # current state is I, but not those who just got updated from A
        stateI = (self.agents['state']==2) & np.logical_not(stateA_evolve)
        newI = (self.agents['state']==2) & (stateA_evolve)
        
        
        new_cases= (newI).sum()
        #new_cases= ((self.agents['state']==2) & (stateA_evolve)).sum()
        
        stateI_index=np.where(stateI)[0]
        # timeI+1
        self.agents['timeI'][stateI_index]+=1
        
        # if time_in_I is reached, move on to the next state
        stateI_evolve= stateI & (self.agents['timeI']>=self.agents['time_in_I'])
        stateI_evolve_index = np.where(stateI_evolve)[0]
        if len(stateI_evolve_index)>0:
           # print('stateI_evolve_index',stateI_evolve_index)
            np.vectorize(self.update_state)(
                agent=stateI_evolve_index,
                new=self.agents['state_after_I'][stateI_evolve_index])
      
        return new_cases

    def run(self,steps=1):
        ''' run model for steps'''
        for i in range(steps):
              self.step()             
                           
    def vaccine(self):
       
        ''' select eligible pool of agents for vaccination '''
        
        # 1) not dead nor currently sick
        state = (self.agents['state']==0) | (self.agents['state']==1) | (self.agents['state']==3)
        # 2) haven't been vaccinated
        not_vaccinated = self.agents['vaccinated']==0
        eli_condition = state & not_vaccinated
        eli_index=np.where(eli_condition)[0]
        eligible_agents = self.agents['ID'][eli_index]
        
        # daily dosage
        num_agents=self.vaccine_plan['total']
        
        num_eligible_agents=eligible_agents.shape[0]
        if num_eligible_agents==0:
            #self.vaccine_all=True
            return
        
        senior_priority=self.vaccine_senior_first
        
        ''' randomly draw samples from pool '''
        if num_agents<=num_eligible_agents:

            if senior_priority: # first put seniors in the list
                print('vac senior first')
                seniors = self.agents['age']== 2 
                eli_seniors_index = np.where(eli_condition & seniors)[0]
                eligible_seniors= self.agents['ID'][eli_seniors_index]
                
                num_eligible_seniors=eligible_seniors.shape[0]
                if num_agents<=num_eligible_seniors:
                    to_vaccinate=np.random.choice(eligible_seniors,size=num_agents,replace=False)
                else:
                    to_vaccinate=eligible_seniors
                    num_agents=num_agents-to_vaccinate.shape[0]
                    eligible_non_seniors_index= np.where(eli_condition & ~seniors)[0] 
                    eligible_non_seniors=self.agents['ID'][eligible_non_seniors_index]
                    to_vac_=np.random.choice(eligible_non_seniors,size=num_agents,replace=False)
                    to_vaccinate=np.hstack((to_vaccinate,to_vac_))
                    np.random.shuffle(to_vaccinate)
            else:
                to_vaccinate=np.random.choice(eligible_agents,size=num_agents,replace=False)
        else:
            to_vaccinate=eligible_agents
        
        m=0
        
        # vaccinate eligible_agents by vaccine type
        for v in self.vaccine_plan:
            
            if v!='total':
                
                if m>len(to_vaccinate):
                    break
                end=m+self.vaccine_plan[v]
                if end>len(to_vaccinate):
                    end=len(to_vaccinate)
                    
                # loop over agents
                for a in to_vaccinate[m:end]:
                    reduction=self.vaccine_reduction[v]
                    self.agents['vaccinated'][a]=1
                    #print('vacc')
                    # if agent is in S
                    if self.agents['state'][a]==0:
                        self.agents['risk_reduction'][a]= reduction
                    else: # if agent is in A
                        prob=np.random.binomial(1,1-reduction)
                        if prob==1:
                            self.update_state(agent=a,new=3)
                m+=self.vaccine_plan[v]

    
    def vaccine_coverage(self,frac1=0.1,frac2=0.1):
        ''' vaccinate frac1+frac2 of all agents in one step '''
        old_plan=self.vaccine_plan
        #num_cover=int(np.round(fraction*self.nPop))
        vac1=int(np.round(frac1*self.nPop))
        vac2=int(np.round(frac2*self.nPop))
        self.vaccine_plan={'total':vac1+vac2,'Pfizer':vac1,'J&J':vac2}
        self.vaccine()
        self.vaccine_plan=old_plan
        return
    
    def agents_df(self):
        '''
            return agent data in dataframe
        '''
        df=pd.DataFrame(self.agents)
        return df
            
    def travel_importation(self,num=10):
        
        ''' import num asymptomatic cases 
        '''
        
        # randomly choose agents to be replaced by imported cases
        importation=np.random.choice(self.agents['ID'],size=num,replace=False) 
      
        for a in importation:
            # scale down importation by travel_scale
            if np.random.rand()>self.travel_scale: continue
           
            self.update_state(agent=a,new=1)
                   
                    
                    
    def plot_compartments(self):
        ''' plot compartment time series'''
        # plotting helper
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
        
    def travel_prep(self):
        ''' read in prevalence data for travel'''
        #self.world_prevalance=pd.read_csv("prevalance_419.csv")
        self.world_prevalance=pd.read_csv("prevalance_84.csv")
        
        
    def travel_simple(self):
        ''' travel implementation
            importations from neighbors based on data prevalence and actual population scale'''
       
        step=str(self.global_day)
        c=self.country
        
        def neighbor_risk(region,weight):
            ''' compute importation risk from neighbors
                depend on the weight array (1/flight hrs between regions)'''
            risk=0
            
            for neighbor in weight[region]: # r is neighbor 
                
                #weight prevalence by population scale
                row=self.world_prevalance[self.world_prevalance.location==neighbor]
                p_r=row[step].item() \
                        *row['actual_pop_scale'].item()
                risk+=weight[region][neighbor]*p_r

            
            return risk
        
        # international risk
        risk_I=neighbor_risk(c,weight_I)
     
        # risk_D per region
        print('region',self.region)
     
        risk=self.params['c_I']*risk_I
        print('international travel',risk_I,'->',risk)
        r=self.region 
        if r.startswith('US.'): #only US has subnational regions and risks
            risk_d=neighbor_risk(r,weight_D)
            risk_D=self.params['c_D']*risk_d
            print('demo travel',risk_d,'->',risk_D)
            risk+=risk_D   
        
        risk=int(np.round(risk))
        risk=min(risk,6)
        risk=max(0,risk)
        print(r,'travel',risk)
        
        # import cases
        self.travel_importation(num=risk)
                
        
        
                
       
        