#!/usr/bin/env python 


import numpy as np 
import matplotlib.pyplot as plt 
from terratools import geographic as geog
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



plume_points = np.loadtxt('geog_plume_points_labels.txt', skiprows=1, comments=None)

plume_number = plume_points[:,3]
lon = plume_points[:,0]
lat = plume_points[:,1]
rad = plume_points[:,2]
temps = plume_points[:,4]
urs = plume_points[:,5]


rho_m = 3300
alpha = 2.5e-5
g = 10

results = []

for pn in np.unique(plume_number):
    print(f'plume number: {pn}')

    plume_index = np.where(plume_number == pn)

    lon_plume = lon[plume_index]
    lat_plume = lat[plume_index]
    rad_plume = rad[plume_index]
    temp_plume = temps[plume_index]
    ur_plume = urs[plume_index]

    # lons, lats, rads = geog.cart2geog(x_plume,y_plume,z_plume)

    plume_fluxes = np.zeros(len(np.unique(rad_plume)))

    plume_results = np.stack([np.unique(rad_plume),plume_fluxes]).T


    for i, r in enumerate(np.unique(rad_plume)):

        lons_rad = np.radians(lon_plume[rad_plume==r])
        lats_rad = np.radians(lat_plume[rad_plume==r])
        temps_rad = temp_plume[rad_plume==r]
        urs_rad = ur_plume[rad_plume==r]
        
        points = np.stack([lats_rad, lons_rad]).T

        x_points = np.copy(points)
        y_points = np.copy(points)

        centre_lo = np.mean(lons_rad)
        centre_la = np.mean(lats_rad)

        x_points[:,0] = centre_la
        y_points[:,1] = centre_lo

        x_distances = haversine_distances([[centre_la, centre_lo]], x_points) * r
        y_distances = haversine_distances([[centre_la, centre_lo]], y_points) * r
        
        points_dist = np.stack([y_distances[0],x_distances[0]]).T
        print(points_dist.shape)

        if points_dist.shape[0] < 3:
            Bp_rad = np.nan
        else:

            A = ConvexHull(points_dist).volume
            T = np.mean(temps_rad)
            u_r = np.mean(urs_rad)

            # print(rho_m, alpha, T, A, u_r, g)

            Bp_rad = rho_m * alpha * T * A * u_r * g

        plume_results[i,1] = Bp_rad

        print(rad, Bp_rad)

    print(plume_results)

    plt.plot(plume_results[:,1], plume_results[:,0], 'o-')
    plt.ylim([3480, 6370])
    plt.show()


