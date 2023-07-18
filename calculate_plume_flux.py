#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import geographic as g
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import haversine_distances
## test ellipse fitting 


def minimum_bounding_ellipse(points):
    hull = ConvexHull(points)
    boundary_points = points[hull.vertices]
    center = np.mean(points, axis=0)
    boundary_points_centered = boundary_points - center
    _, _, v = np.linalg.svd(boundary_points_centered, full_matrices=False)
    direction = v[0]  # First principal component
    angle = np.arctan2(direction[1], direction[0])
    angle_deg = np.degrees(angle)
    rotated_points = np.dot(boundary_points_centered, v.T)
    min_x = np.min(rotated_points[:, 0])
    max_x = np.max(rotated_points[:, 0])
    min_y = np.min(rotated_points[:, 1])
    max_y = np.max(rotated_points[:, 1])
    width = max_x - min_x
    height = max_y - min_y
    a = width / 2
    b = height / 2
    cx = center[0]
    cy = center[1]
    return a, b, cx, cy, angle_deg





# Example usage
# x = np.arange(1,101,1)
# y = (2 * (x + (np.random.rand(100)*100)))


# points = np.stack([x.flatten(), y.flatten()]).T
# print(points)

# a, b, cx, cy, angle_deg = minimum_bounding_ellipse(points)
# print("Semi-major axis (a):", a)
# print("Semi-minor axis (b):", b)
# print("Center (cx, cy):", cx, cy)
# print("Rotation angle (degrees):", angle_deg)

# hull = ConvexHull(points)
# print(hull.volume)
# print(a*b*np.pi)
# plt.figure()
# ax = plt.gca()
# ax.scatter(points[:,0], points[:,1],)
# ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
# ellipse = Ellipse(xy=(cx,cy), width=a*2, height=b*2, angle = angle_deg, 
#                         edgecolor='r', fc='None', lw=2)
# ax.add_patch(ellipse)
# plt.show()



plume_points = np.loadtxt('xyz_plume_points_labels.txt', skiprows=0, comments='')

plume_number = plume_points[:,3]
x = plume_points[:,0]
y = plume_points[:,1]
z = plume_points[:,2]


rho_m = 3300
alpha = 3e-5



for pn in np.unique(plume_number):
    print(f'plume number: {pn}')

    plume_index = plume_number == pn

    x_plume = x[plume_index]
    y_plume = y[plume_index]
    z_plume = z[plume_index]

    lons, lats, rads = g.cart2geog(x_plume,y_plume,z_plume)

    for rad in rads:

        lons_rad = np.radians(lons[rads==rad])
        lats_rad = np.radians(lats[rads==rad])

        points = np.stack([lats_rad, lons_rad]).T

        centre_lo = np.mean(lons_rad)
        centre_la = np.mean(lats_rad)



        distances = haversine_distances([[centre_la, centre_lo]], points) * rad

        

        A = ConvexHull(distances).volume
        T = 3000 # TEMP
        u_r = 5 # TEMP



        Bp_rad = rho_m * alpha * T * A * u_r


