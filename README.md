# terra_analysis

This repository is for my personal version control on scripts to analyse output from the mantle convection modelling code TERRA. Feel free to hack away at any of these codes for your personal use. Most of these rely upon the python module [terratools](https://github.com/mantle-convection-constrained/terratools).

## Script description 

* find_plumes.py - given a TERRA output netcdf file, find the number of plumes with their location. 
* grid_seis_values.py - grid vs values on a basic lon-lat grid in a TERRA model where vs is know at each node point.
* compare_flow_plate_motions.py - compare the flow azimuths in a TERRA model at various depths with plate motions.
* calculate_velocity_gradients.py - using the text files produced by 'grid_seis_values.py', calculate the lateral velocity gradients at each point in the grid.
* calculate_plume_flux.py - Finds the plumes, then takes their temperatures and radial velocities to calculate a plume flux. 
* cluster_hists.py - 
* cluster_input_params.py - 
* downsample_model.py - 
* find_plumes_slabs.py - Similar to find plumes, but also finds slabs in the terra models. 
* get_velocity_grads.sh - 
* make_hist_comp.py - 
* make_hist_temp.py - 
