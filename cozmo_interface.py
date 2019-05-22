#!/usr/bin/env python3

from frame2d import Frame2D
import math

import numpy as np


#TODO find sensible wheel distance parameter (determine experimentally)
wheelDistance = 75.00


# Forward kinematics: compute coordinate frame update as Frame2D from left/right track speed and time of movement
ldtotl = 0
rdtotl = 0


#TODO find sensible noise amplitudes for motor model
cozmoOdomNoiseX = 0.0000003
cozmoOdomNoiseY = 0.0000003
cozmoOdomNoiseTheta = 0.0000006




def cool(left, right, time):
    global ldtotl
    global rdtotl
    lDis = left * time
    rDis = right * time
    ldtotl = ldtotl + lDis
    rdtotl = rdtotl + rDis
    print("\n")
    print("\n")
    print((ldtotl - rdtotl) / (2 * np.pi))
    print("\n")
    print("\n")




def track_speed_to_pose_change(left, right, time):
    # TODO
    lDis = left * time
    rDis = right * time
    theta = (rDis - lDis) / wheelDistance
    x = rDis
    y = 0

    if (abs(theta) > 0.001):
        r = (rDis + lDis) / (2 * theta)
        x = np.sin(theta) * r
        y = (np.cos(theta) - 1) * (-1 * r)
      
    else:
        x = (rDis + lDis) / 2
        y = 0

    return Frame2D.fromXYA(x,y,theta)


# Differential inverse kinematics: compute left/right track speed from desired angular and forward velocity
def velocity_to_track_speed(forward, angular):
    # TODO
   s1 =  forward - (angular * (wheelDistance / 2.0))
   s2 = forward + (angular * (wheelDistance / 2.0))
   print("forward --> ", forward)
   print("angular --> ", angular)
   print("velocity_to_track_speed s1 -->", s1)
   print("velocity_to_track_speed s2 -->", s2)
   return [s1,s2]

# Trajectory planning: given target (ralative to robot frame), determine next forward/angular motion 
# Implement in a linear way±±
# If far away and facing wrong direction: rotate to face target
# If far away and facing target: move forward
# If on target: turn to desired orientation

didhut = False

def target_pose_to_velocity_linear(relativeTarget: Frame2D):
    # TODO
    velocity=0
    angular=0
    global didhut
    x = relativeTarget.mat[0,2]
    y = relativeTarget.mat[1,2]
    d = np.sqrt(((x * x) + (y * y))) 
    theta = math.atan2(y,x)
    print("\n")
    print("theta  -------> ", theta)
    print("Distance from target -> ", d)
    if d > 50 and abs(theta) > 0.1 and not(didhut):
        print("test1")
        print("didhut = -------> ", didhut)
        return  [velocity, theta]

    if d > 50 and not(didhut):
        print("test2")
        return  [0.2 * d, theta]
    else:
        didhut = True
    if didhut:
        dAngular = relativeTarget.angle()
        print("test3")
        print("dAngular - > ", dAngular)
        
        return  [0, dAngular]

    return [velocity, angular]


def target_pose_to_velocity_linearr(relativeTarget: Frame2D):
    # TODO
    velocity=0
    angular=0
    global didhut
    x = relativeTarget.mat[0,2]
    y = relativeTarget.mat[1,2]
    d = np.sqrt(((x * x) + (y * y))) 
    theta = math.atan2(y,x)
    print("\n")
    print("theta ", theta)
    print("Distance from target -> ", d)
    if d > 30 and abs(theta) > 0.1:
        print("test1")
        return  [velocity, theta ]

    if d > 30 and not(didhut):
        print("test2")
        return  [0.5 * d, theta]
    else:
        didhut = True
    if d < 30 or didhut:
        dAngular = relativeTarget.angle()
        print("test3")
        return  [0, dAngular / 5]

    return [velocity, angular]

# Trajectory planning: given target (ralative to robot frame), determine next forward/angular motion 
# Implement by means of cubic spline interpolation


def target_pose_to_velocity_spline(relativeTarget: Frame2D):
    # TODO
    velocity=0
    angular=0
    gin = 10
    global didhut

    x = relativeTarget.mat[0, 2]
    y = relativeTarget.mat[1, 2]

    s = np.sqrt(((x * x) + (y * y)))
    l = np.sqrt(((x * x) + (y * y)))

    t = relativeTarget.mat[1, 2]
    hy = relativeTarget.mat[1,0]

    k = 2 * (3 * t - s * hy) / (s * s)
    #print("Distance from target -> ", l)
    angular = l * k
    velocity = l

    if s > 30 and not(didhut):
       # print("test1")
        return [velocity / gin, angular / gin]
    else:
        didhut = True

    if didhut:
        dAngular = relativeTarget.angle()
       # print("test2")
        return  [0, dAngular / gin]

    return [velocity / gin , angular / gin]


def var_dists(x):
    a = 2.9204*10**-9
    b = -0.00000694006
    c = 0.00226646
    d = 0.265792
    f = -5.54695
    ans = (a * (x**4)) + (b * (x**3)) + (c * (x**2)) + (d * x) + f

    return (ans % 100)

def dists(dist):
    ans = 1.0
    if (dist > 100 and dist < 400):
        ans = 0.001
    elif (dist >= 400 and dist < 510):
        ans = ((dist - 400) / 110) + 0.001
    elif (dist > 50 and dist <= 100):
        ans = 1.0 - (((dist - 50) / 50) + 0.001)

    return 1 - ans

def angles(a,dist):
    ans = 1
    if (abs(a) < math.radians(5) and dist < 5):
        ans = 0.02
    elif (abs(a) < math.radians(5) and dist >= 5 and dist < 40 ):
        ans = 0.99
    elif (abs(a) < math.radians(15) and dist > 9 and dist < 40 ):
        ans = 0.99
    elif (abs(a) < math.radians(20) and dist > 16 and dist < 40):
        ans = 0.99
    elif (abs(a) < math.radians(25) and dist > 22 and dist < 40):
        ans = 0.99
    elif (abs(a) < math.radians(25) and dist > 40 and dist < 45):
        ans = 0.55
    else:
        ans = 0.0001
    return  1 - ans


def cube_sensor_model(trueCubePosition, visible, measuredPosition):
    # print(measuredPosition.angle())
    pointX = trueCubePosition.mat[0, 2]
    pointY = trueCubePosition.mat[1, 2]

    dist = math.sqrt((pointX ** 2) + (pointY ** 2))
    angle = math.atan2(pointY, pointX)

    if (visible):
        mx = trueCubePosition.mult(measuredPosition.inverse())
        e1 = (mx.x() / 20) ** 2
        e2 = (mx.y() / 20) ** 2
        e3 = (mx.angle() / 1.5) ** 2
        p = math.exp(-0.5 * (e1 + e2 + e3))
      #  print(p)
        ans = p # *  dists(dist) * angles(angle,dist)
    else:
        ans =  1 #- (dists(dist) * angles(angle,dist))
    return ans