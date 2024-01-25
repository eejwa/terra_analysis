#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import terra_model as tm
from terratools import geographic as g
from terratools import convert_files
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')



# m = tm.load_model_from_pickle('flow_temp_model.pkl')

try:
    m = tm.read_netcdf(glob.glob('nc*'))
except:
    print(glob.glob('nc*'))
    convert_files.convert(glob.glob('./nc*'))
    m = tm.read_netcdf(glob.glob('nc*'))
# else:
#     print('files messed up leaving')
#     exit()
    

m._surface_radius = 6371
lons_model, lats_model = m.get_lateral_points()
radii = m.get_radii()

print('adding adiabat')
m.add_adiabat()

temps = m.get_field('t')
depths = 6370 - radii


lower_mantle_mask = np.where(depths >= 680)
upper_mantle_mask = np.where(depths <= 680)


bins = np.arange(0,4550,50)

t_hist_depth = np.zeros((len(radii), int(bins.shape[0] -1 )))

for i,depth in enumerate(depths):
    print(depth)
    t_rad = temps[i]
    binned_grads, bin_edges = np.histogram(t_rad.flatten(), bins, density=False)
    t_hist_depth[i] = binned_grads

 
np.save('t_hist_depth.npy', t_hist_depth)
np.save('t_hist_depth_log10.npy', np.log10(t_hist_depth))

lm_hist = t_hist_depth[lower_mantle_mask]
um_hist = t_hist_depth[upper_mantle_mask]

# np.save('grad_hist_depth.npy', grad_hist_depth)
# np.save('grad_hist_depth_log10.npy', np.log10(grad_hist_depth))

np.save('t_hist_lm.npy', lm_hist)
np.save('t_hist_lm_log10.npy', np.log10(lm_hist))

np.save('t_hist_um.npy', um_hist)
np.save('t_hist_um_log10.npy', np.log10(um_hist))


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
c = ax.contourf(bin_edges[:-1],radii,np.log10(t_hist_depth), origin='lower', cmap='autumn_r', vmin=0, vmax=5, bins=20)
ax.set_xlabel('Temp (K)')
ax.set_ylabel('Radius (km)')
plt.colorbar(c, ax=ax)
plt.savefig('2d_hist_temp.pdf')
# # plt.show()
