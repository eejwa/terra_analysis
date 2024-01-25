#!/usr/bin/env python 


import numpy as np
from terratools import terra_model as tm
import terratools.geographic as g
import glob
from sklearn.metrics.pairwise import haversine_distances
from netCDF4 import Dataset




m = tm.read_netcdf(glob.glob('nc*'))

lons_model, lats_model = m.get_lateral_points()
radii_model = m.get_radii()

print(m)

u_xyz = m.get_field('u_xyz')
t = m.get_field('t')
c_hist = m.get_field('c_hist')


tess_4_path = '/Users/earjwara/work/tess_4.txt'
tess_4 = np.loadtxt(tess_4_path)

points_tess_4 = tess_4[:2562,:]


tess_3_path = '/Users/earjwara/work/tess_3.txt'
tess_3 = np.loadtxt(tess_3_path)

points_tess_3 = tess_3[:642,:]






x = points_tess_3[:,1]
y = points_tess_3[:,2]
z = points_tess_3[:,3]

lons_down, lats_down, rs = g.cart2geog(x,y,z)

print(lons_down, lats_down, rs)

print(lons_model, lats_model)

points_model = np.array(list(zip(lons_model, lats_model)))
points_down = np.array(list(zip(lons_down, lats_down)))

print(points_model)
print(points_down)

dists = haversine_distances(np.radians(points_model), np.radians(points_down))

print(dists)
print(dists.shape)

locs = np.argmin(dists, axis=0)

print(locs, locs.shape)

t_down = t[:,locs]
u_down = u_xyz[:,locs]
c_down = c_hist[:,locs]

### Time to make the netcdf file!

nps = len(points_down)
nlayers = len(radii_model)
compositions = 2


ncfile = Dataset('./new.nc',mode='w',format='NETCDF4_CLASSIC')

# create dimensions 

dim_depths = ncfile.createDimension(
            "depths", nlayers
        )

dim_nps = ncfile.createDimension(
            "nps", nps
        )

dim_comps = ncfile.createDimension(
            "compositions", compositions
        )


var_depths = ncfile.createVariable("depths", np.float64, ("depths"))
var_depths.units = "km"
var_depths.radius = 6370
var_depths[:] = 6370 - radii_model

var_vel_x = ncfile.createVariable("velocity_x", np.float64, ("depths", "nps"))
var_vel_x.units = "m/s"
var_vel_x[:] = u_down[:,:,0]

var_vel_y = ncfile.createVariable("velocity_y", np.float64, ("depths", "nps"))
var_vel_y.units = "m/s"
var_vel_y[:] = u_down[:,:,1]

var_vel_z = ncfile.createVariable("velocity_z", np.float64, ("depths", "nps"))
var_vel_z.units = "m/s"
var_vel_z[:] = u_down[:,:,2]

var_temp = ncfile.createVariable("temperature", np.float64, ("depths", "nps"))
var_temp.units = 'K'
var_temp[:] = t_down

var_lat = ncfile.createVariable("latitude", np.float64, ("nps"))
var_lat.units = "degrees"
var_lat[:] = lats_down

var_lon = ncfile.createVariable("longitude", np.float64, ("nps"))
var_lon.units = "degrees"
var_lon[:] = lons_down



var_comp_fracs = ncfile.createVariable(
            "composition_fractions", np.float64, ("compositions", "depths", "nps")
        )

var_comp_fracs[0,:,:] = c_down[:,:,0]
var_comp_fracs[1,:,:] = c_down[:,:,1]
# var_comp_fracs[2,:,:] = c_down[:,:,2]

var_comp_fracs.composition_1_name = "Harzburgite"
var_comp_fracs.composition_1_c = 0
var_comp_fracs.composition_2_name = "Lherzolite"
var_comp_fracs.composition_2_c = 0.2
var_comp_fracs.composition_3_name = "Basalt"
var_comp_fracs.composition_3_c = 1


ncfile.version = 1

ncfile.close()


m2 = tm.read_netcdf(glob.glob('downsampled_model.nc'))
