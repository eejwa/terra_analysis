#!/usr/bin/env python

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

lats_radians = np.radians(lats_unique)
weighting = np.cos(lats_radians)

print(nlons, nlats)

# radii = m.get_radii()
# depths = 6370 - radii

depths = np.loadtxt('depths.txt')
radii = 6371 - depths

fields = ['vp', 'vs', 'density']

vel_grad_bins = np.arange(0.000, 0.005, 0.0001)

print(vel_grad_bins.shape)

# grad_dist_rad = np.zeros((len(radii), vel_grad_bins.shape[0] -1))
gradients_grid = np.zeros((len(radii), int(nlats), int(nlons)))

grad_hist_depth = np.zeros((len(radii), int(vel_grad_bins.shape[0] -1 )))

lower_mantle_mask = np.where(depths >= 680)
upper_mantle_mask = np.where(depths <= 680)

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
    print(grad_mag.shape)
    print(grad_mag.max())

    print(grad_mag[0,:].max())
    print(grad_mag[180,:].max())
    print(grad_mag[-1,:].max())

    grad_mag  = grad_mag * weighting[:, None]

    print(grad_mag[0,:].max())
    print(grad_mag[180,:].max())
    print(grad_mag[-1,:].max())

    # exit()

    # print(grad_lat.shape)
    # print(grad_lon.shape)

    # loop over latitudes 
    
    for w,weight in enumerate(weighting):
        grads_lat = grad_mag[w,:]
        hist_temp, bin_edges = np.histogram(grads_lat, vel_grad_bins, density=False)

        hist_temp = hist_temp.astype(float)
        # weight by over sampling at the poles
        hist_temp *= weight
        if w == 0:
            binned_grads = np.copy(hist_temp)
        else:
            binned_grads = np.add(binned_grads, hist_temp)



    # binned_grads, bin_edges = np.histogram(grad_mag.flatten(), vel_grad_bins, density=False)
    print(binned_grads)
    print(bin_edges.shape)

    gradients_grid[i] = grad_mag
    grad_hist_depth[i] = binned_grads

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(211, projection=ccrs.PlateCarree())
    v = ax.contourf(lons_unique, lats_unique, vss_comp, 15, 
                    cmap='seismic_r', vmin = -1*max_vs, vmax=max_vs)
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

lm_hist = grad_hist_depth[lower_mantle_mask]
um_hist = grad_hist_depth[upper_mantle_mask]

# np.save('grad_hist_depth.npy', grad_hist_depth)
# np.save('grad_hist_depth_log10.npy', np.log10(grad_hist_depth))

np.save('grad_hist_lm.npy', lm_hist)
np.save('grad_hist_lm_log10.npy', np.log10(lm_hist))

np.save('grad_hist_um.npy', um_hist)
np.save('grad_hist_um_log10.npy', np.log10(um_hist))



print('max gradient:', gradients_grid.max())
print('max gradient lower mantle:', gradients_grid[:10,:,:].max())
print('max gradient upper mantle:', gradients_grid[-10:,:,:].max())


# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)
# c = ax.imshow(np.log10(grad_hist_depth), origin='lower', cmap='autumn_r', vmin=0, vmax=8, aspect='auto')
# ax.set_xlabel('$V_{S}$ gradients (km/skm)')
# ax.set_ylabel('Radius (km)')
# plt.colorbar(c,ax=ax)
# plt.show()

#bin_edges[:-1], radii, 
# plt.savefig('2d_hist_velgrad.pdf')
# print('proportion of gradient in reasonable range:', gradients_grid[:10,:,:].max())


# with open('vel_grad_results_local.txt', 'w') as out:
#     out.write('model max_grad_lm \n')
#     out.write(f'{modelname}_{comp} {lower_mantle_grad} \n')


# plt.pcolormesh(grad_dist_rad, bin_edges, radii)
# plt.show()
# print(grad_dist_rad)
