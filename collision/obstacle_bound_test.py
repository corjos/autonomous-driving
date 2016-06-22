# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:10:14 2016

@author: Joseph Corbett-Davies
"""
import matplotlib.pyplot as plt
bds = collision.get_obstacle_bounds()
ori = collision.get_obstacle_orientations()
plt.close('all')
plt.figure()
plt.axes()


obstacle = geometry.box(0,0,2,1)
obstacle = affinity.translate(obstacle, -0.5, -0.5) # shift so back axle at origin
'''
# 17 was a bad one
ori17 = ori[17]
b1 = affinity.rotate(obstacle, ori17[0], origin=(0,0), use_radians=True)
b2 = affinity.rotate(obstacle, ori17[1], origin=(0,0), use_radians=True)

#plt.gca().add_patch(plt.Polygon(np.array(b1.exterior), alpha=0.1))

#plt.gca().add_patch(plt.Polygon(np.array(b2.exterior), alpha=0.1))

union = ops.cascaded_union([b1, b2])
union = geometry.polygon.orient(union).buffer(0.0)#.convex_hull
#union = union.convex_hull
#union = b1.union(b2)
bound = union.convex_hull.buffer(0.0)
bound = geom.polygon.orient(bound).buffer(0.0)

b = collision.obstacle_bound(obstacle, ori17)

plt.gca().add_patch(plt.Polygon(np.array(union.exterior), alpha=0.1, color='r'))
#'''

plt.axis('square')
plt.autoscale()
#'''
for i in range(1,len(bds)):
    bpoly = plt.Polygon(np.array(bds[i].exterior), alpha=0.1)
    plt.gca().add_patch(bpoly)
    txt = plt.text(0,0,str(i))
    plt.autoscale()
    plt.title(str(ori[i]))
    plt.pause(0.5)
    bpoly.remove()
    txt.remove()
#'''