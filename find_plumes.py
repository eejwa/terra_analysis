#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import terra_model as tm
from terratools import geographic as g
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# clustering algorithms
from sklearn.cluster import dbscan

EPSILON =  100 # km
MINPTS = 5

print('reading in model')
m = tm.load_model_from_pickle('flow_temp_model.pkl')
lons_model, lats_model = m.get_lateral_points()
radii = m.get_radii()


print('converting vectors')
m.add_geog_flow()
print('adding adiabat')
m.add_adiabat()
u_geog = m.get_field('u_geog') # lat, lon, rad
temps = m.get_field('t')

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

print('calculating lateral flow magnitude')
u_lat = u_geog[:,:,0]
u_lon = u_geog[:,:,1]
u_rad = u_geog[:,:,2]

u_rad_positive = np.where(u_rad < 0, np.nan, u_rad)
mean_rad_vels = np.nanmean(u_rad_positive) * 2
# mean_temps = np.mean(temps, axis=1)
mean_temps = np.nanpercentile(temps, 95)

mean_rad_vels = np.nanpercentile(u_rad_positive, 95)

print('percentile,', mean_temps)
print('percentile,', mean_rad_vels)
# rad_vel_mask = u_rad > (mean_rad_vels[:, None])
rad_vel_mask = u_rad > (mean_rad_vels)

temp_mask = temps > mean_temps #  (mean_temps[:, None])

# print(temps[1,:], mean_temps[1])

# exit()
u_lat_mag = np.sqrt(np.power(u_lat, 2) + np.power(u_lon, 2))

# find where radial flow velocity is 
# greater than lateral flow velocity
flow_mag_diff = np.divide(u_rad, u_lat_mag)

mag_diff_mask = flow_mag_diff > 1

# points_plumes = xyz_points[np.all([rad_vel_mask, temp_mask], axis = 0)]
# points_plumes_geog = lat_lon_points[np.all([rad_vel_mask, temp_mask], axis = 0)]

points_plumes = xyz_points[np.all([rad_vel_mask, temp_mask], axis = 0)]
points_plumes_geog = lat_lon_points[np.all([rad_vel_mask, temp_mask], axis = 0)]


# points_plumes = xyz_points[temp_mask]



print('finding radii distribution of potential plume xyz points')
radii_plume_points = radii[np.where(flow_mag_diff >= 1)[0]]

# plt.hist(radii_plume_points, bins=20)
# plt.xlabel('Radii (km)')
# plt.ylabel('# potential plume points')
# plt.show()
# exit()

print('percentage of points where radial flow > lateral flow velocity')
print((points_plumes.size / xyz_points.size) * 100, '%')

print('clustering locations to find the number of plumes in the model')
core_samples, labels = dbscan(
    X=points_plumes, eps=EPSILON, min_samples=MINPTS)

print('Number of plumes in the model:')
print(np.max(labels) + 1)

print('% points not clustered')
print(np.sum(labels < 0))
print(np.sum(labels >= 0))
print(labels.size)
print((np.sum(labels < 0) / labels.size) * 100, '%')
print(np.stack([points_plumes[:,0],points_plumes[:,1],
                points_plumes[:,2], labels]).T)
print(np.unique(labels, return_counts=True))
indices_plume = np.where(labels >= 0)

xyz_plume = points_plumes[indices_plume]
xyz_plume_labels = np.c_[xyz_plume * 1000, labels[labels >= 0]]
geog_plume = points_plumes_geog[indices_plume]

# note will be in meters
np.savetxt('xyz_plume_points_labels.txt', xyz_plume_labels, header='x y z label', comments='')
np.savetxt('xyz_plume_points.txt', points_plumes*1000, header='x y z', comments='')

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