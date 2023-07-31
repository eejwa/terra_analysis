#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import terra_model as tm
from terratools import geographic as g

import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')


# clustering algorithms
from sklearn.cluster import dbscan

# clustering parameters 
EPSILON =  100 # km
MINPTS = 5
CUTOFF_RAD = 6071 # km


print('reading in model')
m = tm.load_model_from_pickle('flow_temp_model.pkl')
# m = tm.read_netcdf(glob.glob('nc*'))
m._surface_radius = 6371

lons_model, lats_model = m.get_lateral_points()
radii = m.get_radii()
rad_mask = np.where(radii < CUTOFF_RAD)
radii = radii[rad_mask]


print('converting vectors')
m.add_geog_flow()
print('adding adiabat')
# m.add_adiabat()
u_geog = m.get_field('u_geog')[rad_mask]
temps = m.get_field('t')[rad_mask]
comp_hists = m.get_field('c_hist')[rad_mask]

print(comp_hists)
print(comp_hists.shape)
print(m._c_hist_names)


print(temps.shape)
print(u_geog.shape)

# convert points into xyz
print('converting points into xyz from lat lon rad')
xyz_points = np.zeros((u_geog.shape[0], u_geog.shape[1], 3))
lat_lon_points = np.zeros((u_geog.shape[0], u_geog.shape[1], 3))
for i,rad in enumerate(radii):
    x,y,z = g.geog2cart(lon=lons_model, lat=lats_model, r=rad)
    location = np.stack([x,y,z]).T
    rads = np.ones(lons_model.shape) * rad
    lat_lon_rad_model = np.stack([lons_model, lats_model, rads]).T
    xyz_points[i] = location
    lat_lon_points[i] = lat_lon_rad_model

print(xyz_points.shape)
print(temps.shape)

t_profile = m.mean_1d_profile('t')


print('calculating lateral flow magnitude')
u_lat = u_geog[:,:,0]
u_lon = u_geog[:,:,1]
u_rad = u_geog[:,:,2]
c_bas = comp_hists[:,:,2]
c_harz = comp_hists[:,:,0]
c_lherz = comp_hists[:,:,1]


# change arrays to only have positive or negative radial velocities
u_rad_positive = np.where(u_rad < 0, np.nan, u_rad)
u_rad_negative = np.where(u_rad > 0, np.nan, u_rad)

temps_plume_bound = np.nanpercentile(temps, 95)
temps_slab_bound = np.nanpercentile(temps, 10)

print('upper 5% temps', np.nanpercentile(temps, 95))
print('lower 5% temps', np.nanpercentile(temps, 10))

rad_vels_bound_plume = np.nanpercentile(u_rad_positive, 95)
# i.e. want the most negative
rad_vels_bound_slab = np.nanpercentile(u_rad_negative, 100)

print('upper 5% vel', np.nanpercentile(u_rad_positive, 95))
print('lower 5% vel', np.nanpercentile(u_rad_negative, 5))

# rad_vel_mask = u_rad > (mean_rad_vels[:, None])
rad_vel_mask_plume = u_rad > (rad_vels_bound_plume)
rad_vel_mask_slab = u_rad < (rad_vels_bound_slab)

temp_mask_plume = temps > temps_plume_bound #  (mean_temps[:, None])
temp_mask_slab = temps < temps_slab_bound

mask_comp_bas = c_bas > 0.5

def narrow_down_points(masks, points, geog_points, temps, vels):

    points_feature = points[np.all(masks, axis = 0)]
    points_geog_feature = geog_points[np.all(masks, axis = 0)]
    temps_feature = temps[np.all(masks, axis = 0)]
    urs_feature = vels[np.all(masks, axis = 0)]


    print('percentage of points which are feature')
    print((points_feature.size / points.size) * 100, '%')

    print('clustering locations to find the number of features in the model')
    core_samples, labels = dbscan(
        X=points_feature, eps=EPSILON, min_samples=MINPTS)

    print('Number of features in the model:')
    print(np.max(labels) + 1)

    print('% points not clustered')
    print(np.sum(labels < 0))
    print(np.sum(labels >= 0))
    print(labels.size)
    print((np.sum(labels < 0) / labels.size) * 100, '%')
    print(np.stack([points_feature[:,0],points_feature[:,1],
                    points_feature[:,2], labels]).T)
    print(np.unique(labels, return_counts=True))

    indices_feature = np.where(labels >= 0)

    xyz_feature= points_feature[indices_feature]
    temps_feature = temps_feature[indices_feature]
    urs_feature = urs_feature[indices_feature]
    xyz_feature_labels = np.c_[xyz_feature * 1000, labels[labels >= 0], temps_feature, urs_feature]
    geog_feature = points_geog_feature[indices_feature]


    return points_feature, temps_feature, urs_feature, xyz_feature_labels, geog_feature








# points_plumes = xyz_points[np.all([rad_vel_mask_plume, temp_mask_plume], axis = 0)]
# points_plumes_geog = lat_lon_points[np.all([rad_vel_mask_plume, temp_mask_plume], axis = 0)]
# temps_plumes = temps[np.all([rad_vel_mask_plume, temp_mask_plume], axis = 0)]
# urs_plume = u_rad[np.all([rad_vel_mask_plume, temp_mask_plume], axis = 0)]


# points_plumes = xyz_points[temp_mask]



# print('finding radii distribution of potential plume xyz points')
# radii_plume_points = radii[np.where(flow_mag_diff >= 1)[0]]

# plt.hist(radii_plume_points, bins=20)
# plt.xlabel('Radii (km)')
# plt.ylabel('# potential plume points')
# plt.show()
# exit()

# print('percentage of points which are considered plumes')
# print((points_plumes.size / xyz_points.size) * 100, '%')

# print('clustering locations to find the number of plumes in the model')
# core_samples, labels = dbscan(
#     X=points_plumes, eps=EPSILON, min_samples=MINPTS)

# print('Number of plumes in the model:')
# print(np.max(labels) + 1)

# print('% points not clustered')
# print(np.sum(labels < 0))
# print(np.sum(labels >= 0))
# print(labels.size)
# print((np.sum(labels < 0) / labels.size) * 100, '%')
# print(np.stack([points_plumes[:,0],points_plumes[:,1],
#                 points_plumes[:,2], labels]).T)
# print(np.unique(labels, return_counts=True))

# indices_plume = np.where(labels >= 0)

# xyz_plume = points_plumes[indices_plume]
# temps_plumes = temps_plumes[indices_plume]
# urs_plume = urs_plume[indices_plume]
# xyz_plume_labels = np.c_[xyz_plume * 1000, labels[labels >= 0], temps_plumes, urs_plume]
# geog_plume = points_plumes_geog[indices_plume]



print("Finding plumes")
xyz_plume, temps_plume, urs_plume, xyz_plume_labels, geog_plume = narrow_down_points(masks = [temp_mask_plume,rad_vel_mask_plume, mask_comp_not_bas], 
                                                                                     points = xyz_points, 
                                                                                     geog_points =lat_lon_points, 
                                                                                     temps = temps, 
                                                                                     vels = u_rad_positive)

# note will be in meters
np.savetxt('xyz_plume_points_labels.txt', xyz_plume_labels, header='x y z label temp u_r', comments='')
np.savetxt('xyz_plume_points.txt', xyz_plume*1000, header='x y z', comments='')


print("Finding slabs")
xyz_slab, temps_slab, urs_slab, xyz_slab_labels, geog_slab = narrow_down_points(masks = [temp_mask_slab, rad_vel_mask_slab, mask_comp_bas],
                                                                                points = xyz_points, 
                                                                                geog_points =lat_lon_points, 
                                                                                temps = temps, 
                                                                                vels = u_rad_negative)


np.savetxt('xyz_slab_points_labels.txt', xyz_slab_labels, header='x y z label temp u_r', comments='')
np.savetxt('xyz_slab_points.txt', xyz_slab*1000, header='x y z', comments='')



# cluster_plume_radii = radii[np.where(flow_mag_diff >= 1)[0][indices_plume[0]]]

# plt.hist(cluster_plume_radii, bins=20)
# plt.xlabel('Radii (km)')
# plt.ylabel('# clustered plume points')
# plt.show()


# fig = plt.figure(figsize = (10, 7))
# ax = fig.add_subplot(111)
 
# # Creating plot
# s = ax.scatter(geog_plume[:,0], geog_plume[:,1], c = geog_plume[:,2])
# plt.colorbar(s)

# plt.show()