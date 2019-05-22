import asyncio

import cozmo

from frame2d import Frame2D

async def cozmo_program(robot: cozmo.robot.Robot):
	visible = 0
	cunt = 0
	while True:
		robotPose = Frame2D.fromPose(robot.pose)
		#print("Robot pose: " + str(robotPose))
		cubeIDs = (cozmo.objects.LightCube1Id,cozmo.objects.LightCube2Id,cozmo.objects.LightCube3Id)
		cunt += 1
		for cubeID in cubeIDs:
			cube = robot.world.get_light_cube(cubeID)
			if cube.is_visible:
				visible += 1





		if cunt % 100 == 0:
			cunt = 0
			print("cube visibility visibility ------> ",(visible / 100 ) * 100," %")
			visible = 0
			for cubeID in cubeIDs:
				cube = robot.world.get_light_cube(cubeID)
				if cube.is_visible:
					print("Visible: " + cube.descriptive_name + " (id=" + str(cube.object_id) + ")")

					cubePose = Frame2D.fromPose(cube.pose)

					print("   pose: " + str(cubePose))
					print("   relative pose (2D): " + str(robotPose.inverse().mult(cubePose)))


		await asyncio.sleep(0.05)


cozmo.robot.Robot.drive_off_charger_on_connect = False
cozmo.run_program(cozmo_program, use_3d_viewer=True, use_viewer=True)
