#
# Code to test numpy-stl volume algorithm using an ellipsoid
#
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from matplotlib.colors import LightSource
import numpy as np
import math

pyplot.ion()
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

# Min/max finction from Quickstart page
def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz





s = mesh.Mesh.from_file('plankter.stl')

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(s.vectors,
                                                     shade=True,
                                                     facecolors='white',
                                                     edgecolors='blue',
                                                     alpha=0.15))
# Plot ellipsoid
scale = s.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_aspect('equal')

print('nump-stl calculations:')
# Get min/max dimensions
minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(s)
print('min,max for xyz are: ')
print(minx, maxx, miny, maxy, minz, maxz)

# Calculate volume from stl mesh
volume, cog, inertia = s.get_mass_properties()
print("\nVolume                                  = {0}".format(volume))
print("Position of the center of gravity (COG) = {0}".format(cog))

print('\n\nIdealized geometry calculations:')
# Calculate volume by geometrical properties (ignores descretization)
# The ellipsoid is centered on 0,0,0 so use max's as radii
vol = 4/3*math.pi*maxx*maxy*maxz
print('vol = ',vol)
print('CoG = ',np.zeros(3))

print('\n\nResults from an octave code using the same stl file:')
print('total_volume = ',2.542435320277415e-14)
print('C_gravity = ',np.asarray([-2.509405046241381e-08,
                                 6.410343034598542e-14,
                                 -1.101317119584843e-21]))

