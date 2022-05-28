#!/bin/bash

# to make python script with different job_id
#    so that they can run in parallel 
for i in {0..3}  # inclusive 
do
  # copy file and replace job_id
  sed "s/job_id = 0/job_id = $i/g"  curve_fitting_region.py > sim_single_region_$i.py 

# rm -f sim_single_region_$i.py 
done
~    

