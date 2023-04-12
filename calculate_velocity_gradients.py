#!/usr/bin/env python

from terratools import terra_model as tm
from terratools import geographic
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

m = tm.load_model_from_pickle('seis_model.pkl')

spacing = 0.5
nlons = (360.5 / .5)
nlats = (180.5 / .5)
lons_unique = np.arange(-180, 180.5, 0.5)
lats_unique = np.arange(-90, 90.5, 0.5)

print(nlons, nlats)

radii = m.get_radii()
depths = 6370 - radii
fields = ['vp', 'vs', 'density']

vel_grad_bins = np.arange(0, 0.0101, 0.0001)

print(vel_grad_bins)

grad_dist_rad = np.zeros((len(radii), vel_grad_bins.shape[0] -1))
gradients_grid = np.zeros((len(radii), nlats, nlons))


for i,depth in enumerate(depths):
    print(depth)
    velfile = f'vs_layer_{depth}.dat'
    rad = radii[i]
    dist_per_deg = np.radians(1) * rad
    velarray = np.loadtxt(velfile)
    lons = velarray[:,0]
    lats = velarray[:,1]
    vss = velarray[:,2]
    vss_comp = vss - vss.mean()
    max_vs = np.amax(np.absolute(vss_comp))

    vss = vss.reshape((int(nlats),int(nlons)))
    vss_comp = vss_comp.reshape((int(nlats),int(nlons)))

    # fig = plt.figure(figsize=(6,8))
    # ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
    # v = ax.contourf(lons_unique, lats_unique, vss_comp, 50, cmap='seismic_r',
    #             vmin = -1*max_vs, vmax=max_vs)
    # ax.coastlines()
    # plt.colorbar(v, ax=ax)


    grad_lat, grad_lon = np.gradient(vss, 0.5)
    grad_mag = np.sqrt(np.add(grad_lat**2, grad_lon**2)) / dist_per_deg
    # print(grad_lat.shape)
    # print(grad_lon.shape)
    # binned_grads, bin_edges = np.histogram(grad_mag.flatten(), vel_grad_bins)
    gradients_grid[i] = grad_mag


    # fig = plt.figure(figsize=(8,8))
    # ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())
    # g = ax2.contourf(lons_unique, lats_unique, grad_mag, 20)
    # ax2.coastlines()
    # plt.colorbar(g, ax=ax2)
    # plt.show()
    
#     grad_dist_rad[i] = binned_grads


# grad_dist_rad = np.flip(grad_dist_rad, axis=0)
gradients_grid = np.flip(gradients_grid, axis=0)

print('max gradient:', gradients_grid.max())
print('max gradient lower mantle:', gradients_grid[:10,:,:].max())
print('proportion of gradient in reasonable range:', gradients_grid[:10,:,:].max())


# plt.pcolormesh(grad_dist_rad, bin_edges, radii)
# plt.show()
# print(grad_dist_rad)
