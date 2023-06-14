#!/usr/bin/env python 

from terratools import terra_model as tm
from terratools import geographic
from terratools import convert_files as c
import numpy as np
import time 
import glob

# m = tm.load_model_from_pickle('seis_model.pkl')

try:
    m = tm.read_netcdf(glob.glob('nc*seis'))
except:
    exit()


m.write_pickle('seis_model.pkl')
radii = m.get_radii()
depths = 6370 - radii

print(depths)

of = open('depths.txt', 'w')
for d in depths:
    of.write(f'{d}\n')
of.close()

lons = np.arange(-180, 180.5, 0.5)
lats = np.arange(-90, 90.5, 0.5)

lon_grid, lat_grid = np.meshgrid(lons, lats)

# fields =  ['vp', 'vs', 'density']

fields =  ['vs']



lons_model, lats_model = m.get_lateral_points()

for field in fields:
    values = m.get_field(field)
    for i,rad in enumerate(radii):
        print(rad)
        start = time.time()
        val_mean = values[i].mean()
        depth = 6370 - rad
        filename = f'{field}_layer_{depth}.dat'
        # ilayer1, ilayer2 = tm._bounding_indices(rad, radii)
        with open(filename, 'w') as outfile:
            for lat, lon in zip(lat_grid.flatten(), lon_grid.flatten()):
                # print(lat, lon)
                idx1, idx2, idx3 = m.nearest_indices(lon, lat, 3)
                val_layer1 = geographic.triangle_interpolation(
                lon,
                lat,
                lons_model[idx1],
                lats_model[idx1],
                values[i, idx1],
                lons_model[idx2],
                lats_model[idx2],
                values[i, idx2],
                lons_model[idx3],
                lats_model[idx3],
                values[i, idx3],
            )
                # evaluate vp, vs and density at each point in the grid
                # value = m.evaluate(lon, lat, r=rad, field=field)
                # print(value, val_layer1)
                # value = (val_layer1 - val_mean) * 100

                outfile.write(f'{lon} {lat} {val_layer1}\n')
            end = time.time()
            print(end - start)