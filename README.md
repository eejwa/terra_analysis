# terra_analysis

This repository is for my personal version control on scripts to analyse output from the mantle convection modelling code TERRA. Feel free to hack away at any of these codes for your personal use. Most of these rely upon the python module [terratools](https://github.com/mantle-convection-constrained/terratools).

## Script description 

* compare_flow_plate_motions.py - compare the flow azimuths in a TERRA model at various depths with plate motions.
* calculate_velocity_gradients.py - using the text files produced by 'grid_seis_values.py', calculate the lateral velocity gradients at each point in the grid.
* calculate_plume_flux.py - Finds the plumes, then takes their temperatures and radial velocities to calculate a plume flux. 
* cluster_hists.py - given a collection of histograms in .npy format (see other codes in this directory) cluster the histograms using hierarchical clustering and 
* cluster_input_params.py - Once the clustering is done (see code above), the user can then link these clusters to the varying input parameters (essentially to find which causes the model to change) using mutual information. 
* downsample_model.py - downsamples terra model to a tessellation defined in the code.
* find_plumes_slabs.py - Similar to find plumes, but also finds slabs in the terra models. 
* find_plumes.py - given a TERRA output NetCDF file, find the number of plumes with their location. 
* get_velocity_grads.sh - shell script to automate calculating seismic velocity gradients of several terra models. 
* grid_seis_values.py - grid vs values on a basic lon-lat grid in a TERRA model where vs is known at each node point.
* make_hist_comp.py - for a given terra model, create histograms for each of the compositions with depth.
* make_hist_temp.py - create histograms of temperature vs depth for a given terra model. 
* tess_X.txt - files containing the lats and lons of tessellation points for level X. Used with downsample_model.py.