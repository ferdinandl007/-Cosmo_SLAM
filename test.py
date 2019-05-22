import cozmo

from cozmo.util import degrees, Pose, distance_mm, speed_mmps

import numpy as np

from frame2d import Frame2D
from cozmo_interface import wheelDistance, target_pose_to_velocity_linear, velocity_to_track_speed, \
    track_speed_to_pose_change ,cool

import time


currentPose=Frame2D()


interval = 0.1


arr = []


def recod(acon,robot: cozmo.robot.Robot):
    global currentPose
    while not acon.is_completed:
        lspeed = robot.left_wheel_speed.speed_mmps
        rspeed = robot.right_wheel_speed.speed_mmps
        delta = track_speed_to_pose_change(lspeed, rspeed, interval)
        currentPose = currentPose.mult(delta)
        arr.append([rspeed, lspeed])
        print("trackSpeed" + str([lspeed, rspeed]))
        time.sleep(interval)



def cozmo_update_position(robot: cozmo.robot.Robot):

    for i in range(6):
        acon = robot.drive_straight(distance_mm(50), speed_mmps(50))
        recod(acon,robot)
        acon = robot.turn_in_place(degrees(60))
        recod(acon, robot)
    acon = robot.drive_straight(distance_mm(340), speed_mmps(50))
    recod(acon, robot)
    data = np.array(arr)
    print(arr)
    np.save("hextest10",data)

cozmo.run_program(cozmo_update_position, use_3d_viewer=True)

