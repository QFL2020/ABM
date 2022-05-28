import numpy as np
import pickle 

''' input for travel '''

with open('./ABMpy_region/region_ID.pkl','rb') as f:
    region_ID_from_data=pickle.load(f)
    #{region_list[i]: i for i in range(len(region_list))}


# weight for international travel 
with open('./ABMpy_region/weight_I_norm_apr29.pkl','rb') as f:
    weight_I=pickle.load(f)
    
# weight for domestic travel     
with open('./ABMpy_region/weight_D_norm_apr29.pkl','rb') as f:
    weight_D=pickle.load(f)
    

#world_map={'US':['US.Virginia','US.Maryland','US.Wyoming'],
#           'Mexico':['Mexico'],
#            'Afghanistan':['Afghanistan']}
#country_list=world_map.keys()   
#region_list=[]
#for i in country_list:
#    region_list.extend(world_map[i])
#region_ID_sim={region_list[i]:i for i in range(len(region_list))}

#weight_I={}

#weight_I['US']={'Mexico':0.9,'Afghanistan':0.1}
#weight_I['Mexico']={'US':0.8,'Afghanistan':0.2}
#weight_I['Afghanistan']={'US':0.2,'Mexico':0.1}

# or
#weight_i=np.array([[1,0.9,0.1],[0.8,1,0.2],[0.2,0.1,1]])

#distance

#weight_D={}
#weight_D['US.Virginia']={'US.Maryland':1/3,'US.Wyoming':1/10}
#weight_D['US.Maryland']={'US.Virginia':1/3,'US.Wyoming':1/9}
#weight_D['US.Wyoming']={'US.Virginia':1/10,'US.Maryland':1/9}
#np.array([[1,1/10,1/3],
#                [1/10,1,1/4],[1/3,1/4,1]])
#weight_D['Canada']=np.array([[1,1/2],[1/2,1]])
#weight_D['Afghanistan']=np.array([[1]])
