#!/usr/bin/env python3

# Copyright (c) 2019 Matthias Rolf, Oxford Brookes University

'''

'''

import cozmo

from cozmo.util import degrees, Pose, distance_mm, speed_mmps

import numpy as np

from frame2d import Frame2D
from cozmo_interface import wheelDistance, target_pose_to_velocity_linear,velocity_to_track_speed,track_speed_to_pose_change ,didhut

import time



currentPose=Frame2D()

# TODO allow the target to be chosen as console parameter
targetPose=Frame2D.fromXYA(250,150,3)

interval = 0.1

def cozmo_drive_to_target(robot: cozmo.robot.Robot):
    global currentPose
    didhut = False
    while True:
        relativeTarget = Frame2D() # TODO determine current position of target relative to robot
        relativeTarget = currentPose.inverse().mult(targetPose)

        print("relativeTarget"+str(relativeTarget))
        velocity = target_pose_to_velocity_linear(relativeTarget)
        print("velocity"+str(velocity))
        trackSpeed = velocity_to_track_speed(velocity[0],velocity[1])
        print("trackSpeedCommand"+str(trackSpeed))
        lspeed = robot.left_wheel_speed.speed_mmps
        rspeed = robot.right_wheel_speed.speed_mmps
        print("trackSpeed"+str([lspeed, rspeed]))
        delta = track_speed_to_pose_change(lspeed, rspeed,interval)
        currentPose = delta.mult(currentPose)
        print("currentPose"+str(currentPose))
        print()
        robot.drive_wheel_motors(l_wheel_speed=trackSpeed[0],r_wheel_speed=trackSpeed[1])
        time.sleep(interval)


cozmo.run_program(cozmo_drive_to_target,use_3d_viewer=True,use_viewer=True)

