import requests
import numpy as np
import pandas as pd
import pickle
import time
import sys

''' obtain the minimum flight hours bewteen one country and the rest'''

# iata codes for countries 
with open('country_iata.pkl','rb') as f:
    iata=pickle.load(f)
    
    
# current job 
job_id = 0
print('job id',job_id)

# tequila API
url= \
"https://tequila-api.kiwi.com/v2/search?fly_from=JFK&fly_to=IAD&\
date_from=01%2F5%2F2021&date_to=31%2F12%2F2021&\
&adults=1\
&only_working_days=false&only_weekends=false&partner_market=us\
&vehicle_type=aircraft&limit=400"

def flight_hr(origin,destination,url=url):
    ''' obtain the minimum flight hours between origin and destination'''
    
    i_o=url.find('fly_from=')+9
    i_t=url.find('fly_to=')+7
    url=url[:i_o]+origin+url[i_o+3:]
    url=url[:i_t]+destination+url[i_t+3:]
    print(origin,destination)

    z={'error_code':429}

    while 'error_code' in z.keys() and z['error_code']== 429: # keep trying if can't connect 
        print('connecting to API...')
        try:
            z=requests.get(url, headers={'apikey':'h9SPJMYOmyBUKErHTUpHls71cZtqXqly'}).json()
        except:
            print('some Error')
            time.sleep(5)
            continue
            
    # if no data        
    if 'data' not in z.keys() or len(z['data'])==0:
        print('no flight')
        print(z)
        return 10**6
    
    print('results len',len(z['data']))
    
    # convert to hours and find minimum  
    durations=[]
    for i in range(len(z['data'])):
        #h,m=z['data'][i]['duration']
        #durations.append(int(h)+int(m[:-1])/60.)
        durations.append(z['data'][i]['duration']['departure']/3600)
    durations=np.array(durations)
    print(durations.mean())
    print(durations.min())
    return durations.min()

# the country for this job 
c1=list(iata.keys())[job_id]

    
weight={c1:{}}
for c2 in iata: # loop over all the rest of countries 
    hrs=10**6
    if (c2==c1) : continue
    #or (c2 in weight[c1].keys() and weight[c1][c2]<hrs): continue
    for o in iata[c2]:
        for d in iata[c1]:
            print(c2,c1)
            hrs=min(hrs,flight_hr(o,d))      
    weight[c1][c2]=np.round(hrs,2)
    
# save 
with open('flight_hrs_4/country_'+str(job_id)+'.pkl','wb') as f:
    pickle.dump(weight,f)
