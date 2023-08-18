#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import terra_model as tm
from terratools import geographic as g
from terratools import convert_files
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')



m = tm.load_model_from_pickle('flow_temp_model.pkl')
# m = tm.read_netcdf(glob.glob('nc*'))
m._surface_radius = 6371
lons_model, lats_model = m.get_lateral_points()
radii = m.get_radii()

print('adding adiabat')
m.add_adiabat()

c_hist = m.get_field('c_hist')
c_names = m._c_hist_names
depths = 6370 - radii

print(c_names)

lower_mantle_mask = np.where(depths >= 680)
upper_mantle_mask = np.where(depths <= 680)


bins = np.arange(0,1,0.01)


for j,c_name in enumerate(c_names):
    t_hist_depth = np.zeros((len(radii), int(bins.shape[0] -1 )))
    total_c_depth = np.zeros((len(radii), 3))
    c_name = c_name.lower()

    for i,depth in enumerate(depths):
        c_rad = c_hist[i,:,j]

        total_c = np.median(c_rad)
        c_25 = np.quantile(c_rad, .25)
        c_75 = np.quantile(c_rad, .75)
        total_c_depth[i] = np.array([c_25, total_c, c_75])

        print(depth)

        binned_grads, bin_edges = np.histogram(c_rad.flatten(), bins, density=False)
        t_hist_depth[i] = binned_grads

        
    np.save(f'{c_name}_hist_depth.npy', t_hist_depth)
    np.save(f'{c_name}_hist_depth_log10.npy', np.log10(t_hist_depth))

    lm_hist = t_hist_depth[lower_mantle_mask]
    um_hist = t_hist_depth[upper_mantle_mask]

    # np.save('grad_hist_depth.npy', grad_hist_depth)
    # np.save('grad_hist_depth_log10.npy', np.log10(grad_hist_depth))

    np.save(f'{c_name}_hist_lm.npy', lm_hist)
    np.save(f'{c_name}_hist_lm_log10.npy', np.log10(lm_hist))

    np.save(f'{c_name}_hist_um.npy', um_hist)
    np.save(f'{c_name}_hist_um_log10.npy', np.log10(um_hist))


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    c = ax.contourf(bin_edges[:-1],radii,np.log10(t_hist_depth), origin='lower', cmap='autumn_r', vmin=0, vmax=5, bins=20)
    ax.set_xlabel(f'{c_name} fraction')
    ax.set_ylabel('Radius (km)')
    plt.colorbar(c, ax=ax)
    plt.show()
    plt.savefig(f'2d_hist_{c_name}.pdf')


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    c = ax.plot(total_c_depth[:,0],radii, '-o')
    c = ax.plot(total_c_depth[:,1],radii, '-o')
    c = ax.plot(total_c_depth[:,2],radii, '-o')

    ax.set_xlabel(f'{c_name} total fraction')
    ax.set_ylabel('Radius (km)')
    plt.show()
    # plt.savefig(f'2d_hist_{c_name}.pdf')