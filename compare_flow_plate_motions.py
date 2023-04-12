#!/usr/bin/env python


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import terra_model as tm
import matplotlib.pyplot as plt
from scipy.stats import circmean
from global_land_mask import globe 


# plt.style.use('ggplot')
def diff(angles1, angles2):
  res = np.remainder(np.absolute(np.subtract(angles1, angles2)), 360)

  res = np.where(res > 180, 360 - res, res)
  
  return res

m = tm.load_model_from_pickle('flow_temp_model.pkl')

lons_model, lats_model = m.get_lateral_points()
radii = m.get_radii()
m.add_geog_flow()
u_geog = m.get_field('u_geog') # lat, lon, rad

ocean_mask = globe.is_ocean(lats_model, lons_model)
land_mask = globe.is_land(lats_model, lons_model)

u_lat = u_geog[:,:,0]
u_lon = u_geog[:,:,1]
u_rad = u_geog[:,:,2]

az = np.degrees(np.arctan2(u_lon, u_lat))
az[az<0] += 360

az_surface = az[-1,:]

print(az_surface)
print(az[-2,:])
print(diff(az_surface, az[-2,:]))
print('')
print(diff(az_surface, az[0,:]))
print(az_surface)
print(az[0,:])
az_diffs = np.zeros(az.shape)
az_diff_ocean = np.zeros((az.shape[0], np.sum(ocean_mask)))
az_diff_land =  np.zeros((az.shape[0], np.sum(land_mask)))

for i in range(len(radii)):
    print(i)
    az_diffs[i] = diff(az_surface, az[i,:])
    az_diff_ocean[i] = diff(az_surface[ocean_mask], az[i,:][ocean_mask])
    az_diff_land[i] = diff(az_surface[land_mask], az[i,:][land_mask])



mean_az_diffs = np.degrees(circmean(np.radians(az_diffs), axis=1))
mean_az_diff_ocean = np.degrees(circmean(np.radians(az_diff_ocean), axis=1))
mean_az_diff_land = np.degrees(circmean(np.radians(az_diff_land), axis=1))


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.plot(mean_az_diffs, radii, label='all points', c = 'C0')
ax.plot(mean_az_diff_ocean, radii, label='ocean points', c = 'C1')
ax.plot(mean_az_diff_land, radii, label='land points', c = 'C2')
ax.set_xlabel("Angular difference of flow ($^\circ$)")
ax.set_ylabel('Radius (km)')
ax.axhline((6371 - 200), 0, np.amax(mean_az_diffs), label='200 km', c = 'C3')
ax.axhline((6371 - 410), 0, np.amax(mean_az_diffs), label='410 km', c = 'C4')
ax.axhline((6371 - 660), 0, np.amax(mean_az_diffs), label='660 km', c = 'C5')
ax.legend()
plt.tight_layout()
plt.savefig('flow_azimuth_depth_BB033.pdf')
plt.show()


