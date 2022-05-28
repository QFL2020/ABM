Goal: to find ABM models that best fit data, input different vaccine scenarios to the models and project them forward in time


* tutorial.ipynb: basic usage of  the ABMpy_region library




Simulation Procedure:
* 0) for a grid of parameter combinations, run simulations with default setting for all the combinations, store simulation time series in a data frame
* 1) for each region, find the best-fit simulation and parameter combination to their case data, the case data start from the day where the region reached 100 cases/million, and end at 5/31/2021 or the day where vaccine coverage reached 3% 
* 2) for each region, with the best-fit parameter combination, evolve a new ABM model to a desired end date
* 3) for each region, continue to evolve their ABM model, but turn on travel, demographic specification and vaccination, and refit S2A
* 4) connect all the regional models into one global model, simulate travel, add vaccine donation and project forward 


      0)  param_explore.py


* Explore parameter grid, output correspondence between input parameter and simulated time series.
* Input: fixed parameters, a grid of variable parameters, number of jobs in total, number of jobs per CPU, output path 
* Output: parameters and simulated data (pickle) 


        Param_explore_save_csv.py
* Save grid search sim to a dataframe 


        prepare_data.ipynb 
* Download and process case data time series 
* Convert to prevalence data: case/population
* Generate both daily new and cumulative dataframes
* Generate vaccine dosage dataframe for global and US states 
* Prepare a dataframe for regional fitting (fitting_info_aug3.csv) to include start and end days of first fitting period 


1. Curve_fitting_region.py
find the best-fit sim and parameter set for a region from the grid search, simple fitting without vaccination, travels or demographic differences between regions
Input: region ID, case data, fitting_info
Output: fit_job_id.pkl 
the best-fit parameters and additional info including ‘group_size','gamma','gamma_M','S2A', 'min_sim_number' (random seed’) and min RMSE of best-fit sim


2. Curve_fitting_before_vac.ipynb
- visualizing regional curving fitting results (without vaccination) 
- can save to pdf 
- compile regional results (fit_job_id.pkl) to a single dataframe, e.g. “curve_fitting_results_83_before_vaccine.csv”


3.  sim_single_region_before_vac.py
* Evolve regional model to desired end date 
* Input: 
   * regional fit results (parameters and random seed number)
* Output: regional ABM object (path/abm_region_ID.pkl) 


4.  sim_single_region_after_vac.py
* continue to evolve regional models independently
  - update regional demographics 
  - add vaccination
  - add effect of travels, from data 
  - adjust S2A and find best-fit 
             - input:  
-  regional ABM object from sim_single_region_before_vac.py
- Vaccine dosage for regions
- Case data 


             - output : regional ABM object  and plots of time series with different S2A


5. meta_pop_project_*.py
* combine regional models into one meta model and project forward
            - simulate travels
            - simulate different vaccine scenario
     -     input:  
- regional ABM object from sim_single_region_after_vac.py
- first period fitting results curve_fitting_results_83_before_vaccine.csv
- vaccine plan for specific scenario
     -     output:
        Time series data for all regions, and their actual_pop_scale 


Example script: 
Meta_pop_project_to_sep_20percent_base.py, for the 20percent coverage baseline 
Meta_pop_project_priority_feature.py, for chosen feature to be prioritized 
         




Additional data collection processing scripts:


* vaccine_X%_compute.ipynb
   * Compute regional vaccine dosage needed to achieve a given coverage in 3 months
   * Input: 
      * time series of baseline 3-month projection ( ‘meta_base_data.pkl')
      * Target coverage 
   * Output:
      * Vaccine plan for countries achieving the target coverage ("vaccine_info_83_"+str(int(target_percentage))+"percent_plan.csv")
      * Donation needed to achieve target_percentage 


* Vaccine_plan_surging.ipynb
* Compute vaccine plan for surging countries, prioritized to chosen features 
* Input:
   * Vaccine dosage data
   * Case data
   * time series of baseline 3-month projection
   * Global donation dosage
   * Feature of interest
* Output:
   * Vaccine donation plan distributed to priorized the feature (absolute dosage, name: "vaccine_plan_priority_"+feature+"_83.csv") 




* Flight_hours_and_code.ipynb
   * Example of how to obtain the minimum flight hours between regions from Tequila-API
   * get the iata code of all the airports in a country's captial 
   * Input: country-list.csv, iso_country_code.csv,  airports.csv 
   * Output: country-iata dictionary (country_iata.pkl)
* Flights.py
   * obtain the minimum flight hours bewteen one country and all other countries 
   * Input: country_iata.pkl
   * Output: a dictionary contains the minimum hours in the form of weight[country][neighbor] (flight_hrs_4/country_'+str(job_id)+'.pkl') 
* Travel_weight_matrix.ipynb
   * Generate weight matrix for international travels and US domestic travels 
   * Output: 
      * weight_I_norm_apr29.pkl
      * weight_D_norm_apr29.pkl
* 



Data:
region_list.pkl
fitting_info_aug3.csv
vaccineRateDaily_global_20210630.csv
vaccineRateDaily_USA_20210630.csv 
vaccine_info_83.csv  


Slurm_job_
sbatch -c 2 -t 24:0:0 -p lrgmem jupyter_notebook_start




class ABM(builtins.object)
 |  ABM(params=None)
 |  
 |  regional model
 |  
 |  Methods defined here:
 |  
 |  A2I_or_R(self, agent=0)
 |      decide the outcome of agents entering A
 |  
 |  I2R_or_D(self, agent=0)
 |      decide the outcome of agents entering I
 |  
 |  __init__(self, params=None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  agents_df(self)
 |      return agent data in dataframe
 |  
 |  agents_evolve(self)
        Time +1 
 |      evolve all agents one step forward
 |  
 |  demo_reset(self, demo_dist=None, case_fatality=None, case_frac=None)
 |      input: numpy arrays of length 12 to specify the value of each demographic group
 |      
 |      demo_dist: demographic proportion of each demo group
 |      case_fatality: likewise
 |      case_frac: the proportion of each demo group in all cases
 |      
 |      this is a function for 
 |      - resetting demographic distribution
 |      - adjustingS2A for each demo group such that the overall S2A of the region stays the same
 |      - resetting agents' demographic attributes to be consistent with new demo_dist, case_fatality and case_frac
 |  
 |  make_mixing_array_np(self)
 |      generate mixing array (to speed up meetings) at the region level and at the group
 |      for group level mixing array, only update when 
 |          region_group_need_update_mixing[region][group] = True
 |  
 |  meeting(self)
 |      loop over all infectious agents to initiate meetings
 |  
 |  one_v_many_meetings_on(self, agent)
 |      agent meents other agents randomly O(n) verion
 |  
 |  one_v_one_meeting(self, agent1=0, agent2=1)
 |      meeting between agent 1&2
 |      1 is infectious
 |  
 |  plot_compartments(self)
 |      plot compartment time series
 |  
 |  plot_data(self, col=['new_cases'])
 |      plot just one column from data frame
 |  
 |  run(self, steps=1)
 |      run model for steps
 |  
 |  step(self)
 |      daily evolution
 |      1) meeting
 |      2) vaccination
 |      3) update agents by state  (agents_evolve)
 |      4) update dataframe
 |  
 |  travel_importation(self, num=10)
 |      import num*self.travel_scale  asymptomatic cases
 |  
 |  travel_prep(self)
 |      read in prevalence data for travel
 |  
 |  travel_simple(self)
 |      travel implementation
 |      importations from neighbors based on data prevalence and actual population scale
 |  
 |  update_compartments(self, new_cases=None)
 |      helper to update the compartment dataframe
 |  
 |  update_region_group_mixing(self, agent=0)
 |      set region_group_need_update_mixing[region][group]=True
 |      so mixing array for the group the input agent is in will get updated in next step
 |  
 |  update_state(self, agent=0, new=1)
 |      update state of an agent
 |  
 |  update_travel_scale(self, scale)
 |      helper to update travel scale
 |  
 |  vaccine(self)
 |      select eligible pool of agents for vaccination
 |  
 |  vaccine_coverage(self, frac1=0.1, frac2=0.1)
 |      vaccinate frac1+frac2 of all agents in one step
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)




Global Model:


class Meta_ABM(builtins.object)
 |  Meta_ABM(ABM=[])
 |  
 |  global model
 |  consists of a list of individual ABM, handles their travels
 |  
 |  Methods defined here:
 |  
 |  __init__(self, ABM=[])
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  plot_compartments(self)
 |      plot compartment time series
 |  
 |  plot_data(self, col=['new_cases'])
 |      plot just one column from data frame
 |  
 |  run(self, steps=1, multiprocessing=False)
 |      run model for steps
 |  
 |  step(self, multiprocessing=False)
 |      daily evolution
 |      1) travel
 |      2) run regional models
 |      3) update dataframe
 |  
 |  travel_replace(self)
 |      NOT USED
 |      simulate travels from j->i 
 |      by replacing attributes of agents in i by attributes of agents in j, 
 |      without actually moving agents from j to i
 |      conserving number of agents in all regions
 |  
 |  travel_simple(self)
 |      travel
 |       importations from neighbors based on simulated prevalence and actual population scale
 |      - calculate coutry level prevelance
 |      - scale to actual population size
 |  
 |  update_compartments(self)
 |      keeps record of global counts
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)


























----------------------------------------------------------------------------------------------------------------------------
Bash code to make an array of scripts 


#!/bin/bash


# to make python script with different job_id
#    so that they can run in parallel 
for i in {0..3}  # inclusive 
do
  # copy file and replace job_id
# replace “job_id = 0” in curve_fitting_region.py with “job_id = $i$ and copy the rest to sim_single_region_$i.py
  sed "s/job_id = 0/job_id = $i/g"  curve_fitting_region.py > sim_single_region_$i.py 
# removing files
# rm -f sim_single_region_$i.py 
done


Job_id in python scripts is a variable to specify different input, e.g. different regions, different target vaccine plane etc.  
----------------------------------------------------------------------------------------------------------------------------


SLURM
Blue is example code, explanation in black 
#!/bin/bash  Starting a bash script 


#SBATCH --job-name=surging 
#SBATCH --time=10:0:0  job time limit
#SBATCH --partition=shared requisition partition 
Space after # means comments 
# SBATCH --partition=lrgmem
# SBATCH --partition=parallel


#SBATCH -N 1 number of node
# SBATCH --ntasks-per-node=28


# number of tasks (processes) per node
# SBATCH --ntasks-per-node=2
# SBATCH --mem-per-cpu=4900 if specifying memory
#SBATCH --mem=112640
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=all to receive notification
#SBATCH --mail-user=yhuang98@jhu.edu
#SBATCH --output=out/job-%J.out path to system output file


Run an array of scripts 
#SBATCH --array=0-3
 


#### load and unload modules on MARCC you may need
#### activate the right environment
ml anaconda
conda activate /home-net/home-4/yhuang98@jhu.edu/code/ABMpy/env
## run python $SLURM_ARRAY_TASK_ID goes from 0-3
python -u sim_single_region_$SLURM_ARRAY_TASK_ID.py


echo "Finished with job $SLURM_JOBID"
#echo "Finished with job $SLURM_ARRAY_TASK_ID"