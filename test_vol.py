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

def get_mass_props2(mesh=None):
    normals = mesh.get_unit_normals()
    areas = mesh.areas
    total_area = areas.sum()
    centroids = mesh.centroids
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center

def get_mass_props4(mesh=None):
    normals = mesh.get_unit_normals()
    s = np.sign(np.sum(mesh.centroids*normals,axis=1))
    normals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)
    areas = mesh.areas
    total_area = areas.sum()
    centroids = mesh.centroids
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center

def get_mass_props3(mesh=None):
    normals = mesh.get_unit_normals().astype('float64')
    areas = mesh.areas.astype('float64')
    total_area = areas.sum()
    centroids = mesh.centroids.astype('float64')
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center

def get_mass_props5(mesh=None):
    normals = mesh.get_unit_normals().astype('float64')
    s = np.sign(np.sum(mesh.centroids*normals,axis=1))
    normals *= s.reshape([s.shape[0],1]).repeat(3,axis=1)
    areas = mesh.areas.astype('float64')
    total_area = areas.sum()
    centroids = mesh.centroids.astype('float64')
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center


#==============================================================================
# Code to find the intersection, if there is one, of a line and a triangle
# in 3D, due to @Jochemspek,
# https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
def intersect_line_triangle(q1,q2,p1,p2,p3):
    def signed_tetra_volume(a,b,c,d):
        return np.sign(np.dot(np.cross(b-a,c-a),d-a)/6.0)

    s1 = signed_tetra_volume(q1,p1,p2,p3)
    s2 = signed_tetra_volume(q2,p1,p2,p3)

    if s1 != s2:
        s3 = signed_tetra_volume(q1,q2,p1,p2)
        s4 = signed_tetra_volume(q1,q2,p2,p3)
        s5 = signed_tetra_volume(q1,q2,p3,p1)
        if s3 == s4 and s4 == s5:
            n = np.cross(p2-p1,p3-p1)
            #t = -np.dot(q1,n-p1) / np.dot(q1,q2-q1)
            t = np.dot(p1-q1,n) / np.dot(q2-q1,n)
            return q1 + t * (q2-q1)
    return None

def count_intersections(mesh=None,ref_point=np.ones(3),project=1.e-6):
    normals = mesh.get_unit_normals()
    centroids = mesh.centroids
    test_points = centroids + project*normals
    m = normals.shape[0]
    counts = np.zeros([m,1])
    for i in range(m):
        q1=ref_point
        q2 = test_points[i]
        for j in range(m):
            vecs = mesh.vectors[j,:,:]
            p1 = vecs[0,:]
            p2 = vecs[1,:]
            p3 = vecs[2,:]
            ilt = intersect_line_triangle(q1,q2,p1,p2,p3)
            if ilt is not None:
                counts[i] += 1
    return counts

    
def get_mass_props6(mesh=None):
    normals = mesh.get_unit_normals()
    counts = count_intersections(mesh=mesh)
    evens = counts % 2==0
    odds = counts % 2!=0
    S = np.zeros(counts.shape)
    S[odds] = -1
    S[evens] = 1
    normals *= S.reshape([S.shape[0],1]).repeat(3,axis=1)
    areas = mesh.areas
    total_area = areas.sum()
    centroids = mesh.centroids
    volumes = areas*(centroids*normals).sum(axis=1).reshape([areas.shape[0],1])/3
    total_volume = volumes.sum()
    tet_centroids = 0.75 * centroids
    volume_center = (tet_centroids*volumes.repeat(3,axis=1)).sum(axis=0)/total_volume
    return total_area,total_volume,volume_center,counts,S


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
print('Total area = ',s.areas.sum())

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
print('total_volume = ',4.911965863450262e-09)

# Calculate volume from stl mesh with get_mass_props2 function
print('\n\nget_mass_props2 calculations:')
total_area,total_volume,volume_center = get_mass_props2(mesh=s)
print("\ntotal_volume = {0}".format(total_volume))
print("volume_center = {0}".format(volume_center))
print("total_area = {0}".format(total_area))

# Calculate volume from stl mesh with get_mass_props3 function
print('\n\nget_mass_props3 calculations:')
total_area,total_volume,volume_center = get_mass_props3(mesh=s)
print("\ntotal_volume = {0}".format(total_volume))
print("volume_center = {0}".format(volume_center))
print("total_area = {0}".format(total_area))

# Calculate volume from stl mesh with get_mass_props4 function
print('\n\nget_mass_props4 calculations:')
total_area,total_volume,volume_center = get_mass_props4(mesh=s)
print("\ntotal_volume = {0}".format(total_volume))
print("volume_center = {0}".format(volume_center))
print("total_area = {0}".format(total_area))

# Calculate volume from stl mesh with get_mass_props5 function
print('\n\nget_mass_props5 calculations:')
total_area,total_volume,volume_center = get_mass_props5(mesh=s)
print("\ntotal_volume = {0}".format(total_volume))
print("volume_center = {0}".format(volume_center))
print("total_area = {0}".format(total_area))

# Calculate volume from stl mesh with get_mass_props6 function
print('\n\nget_mass_props6 calculations:')
total_area,total_volume,volume_center,counts,S = get_mass_props6(mesh=s)
print("\ntotal_volume = {0}".format(total_volume))
print("volume_center = {0}".format(volume_center))
print("total_area = {0}".format(total_area))
