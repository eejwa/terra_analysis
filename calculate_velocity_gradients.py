#!/usr/bin/env python

from terratools import terra_model as tm
from terratools import geographic
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import os 

cwd = os.getcwd()
other_path, current_dirname = os.path.split(cwd)
parent_dir = os.path.split(other_path)[1]
print(parent_dir)
modelname = parent_dir.split('_NC_')[0]
comp = parent_dir.split('_')[-1]

print(modelname)

# m = tm.load_model_from_pickle('seis_model.pkl')

spacing = 0.5
nlons = (360.5 / .5)
nlats = (180.5 / .5)
lons_unique = np.arange(-180, 180.5, 0.5)
lats_unique = np.arange(-90, 90.5, 0.5)

print(nlons, nlats)

# radii = m.get_radii()
# depths = 6370 - radii

depths = np.loadtxt('depths.txt')
radii = 6371 - depths

fields = ['vp', 'vs', 'density']

vel_grad_bins = np.arange(0, 0.0101, 0.0001)

print(vel_grad_bins)

grad_dist_rad = np.zeros((len(radii), vel_grad_bins.shape[0] -1))
gradients_grid = np.zeros((len(radii), int(nlats), int(nlons)))


for i,depth in enumerate(depths):
    print(depth)
    velfile = f'vs_layer_{depth}.dat'
    rad = radii[i]
    dist_per_deg = np.radians(1) * rad
    velarray = np.loadtxt(velfile)
    lons = velarray[:,0]
    lats = velarray[:,1]
    vss = velarray[:,2]
    # replace 0 values with mean 

    vss_comp = ((vss - vss.mean()) / vss.mean()) *100


    vss = vss.reshape((int(nlats),int(nlons)))
    vss_comp = vss_comp.reshape((int(nlats),int(nlons)))

    vss[vss_comp < -20] = np.median(vss)
    vss[vss_comp > 20] = np.median(vss)

    vss_comp[vss_comp < -20] = 0
    vss_comp[vss_comp > 20] = 0
    max_vs = np.amax(np.absolute(vss_comp))

    grad_lat, grad_lon = np.gradient(vss, 0.5)
    grad_mag = np.sqrt(np.add(grad_lat**2, grad_lon**2)) / dist_per_deg
    print(grad_mag.max())
    # print(grad_lat.shape)
    # print(grad_lon.shape)
    # binned_grads, bin_edges = np.histogram(grad_mag.flatten(), vel_grad_bins)
    gradients_grid[i] = grad_mag

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
    v = ax.contourf(lons_unique, lats_unique, vss_comp, 15, cmap='seismic_r',
                vmin = -1*max_vs, vmax=max_vs)
    ax.coastlines()
    plt.colorbar(v, ax=ax, label='V$_{S}$ (%)')
    ax.set_title(f'Depth {depth} km')

    ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())
    g = ax2.contourf(lons_unique, lats_unique, grad_mag, 15)
    ax2.coastlines()
    plt.colorbar(g, ax=ax2, label='V$_{S}$ gradient (kms$^{-1}$km$^{-1}$)')
    plt.savefig(f'seisvel_grads_{depth}.pdf')
    # plt.show()
    plt.close()
    
#     grad_dist_rad[i] = binned_grads

# grad_dist_rad = np.flip(grad_dist_rad, axis=0)
# gradients_grid = np.flip(gradients_grid, axis=0)
lower_mantle_grad = gradients_grid[:10,:,:].max()
print('max gradient:', gradients_grid.max())
print('max gradient lower mantle:', gradients_grid[:10,:,:].max())
print('max gradient upper mantle:', gradients_grid[-10:,:,:].max())
# print('proportion of gradient in reasonable range:', gradients_grid[:10,:,:].max())


with open('vel_grad_results_local.txt', 'w') as out:
    out.write('model max_grad_lm \n')
    out.write(f'{modelname}_{comp} {lower_mantle_grad} \n')


# plt.pcolormesh(grad_dist_rad, bin_edges, radii)
# plt.show()
# print(grad_dist_rad)
