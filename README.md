# PrediKCS

Example ROS package for converting desired end-effector motion in Cartesian space to joint velocities for redundant robotic manipulators. This package implements the Predictive Kinematic Control Search (PrediKCS) algorithm for converting desired Cartesian velocity to joint velocities on a URDF-specified robotic arm with 7+ degrees of freedom. While the algorithm itself does not depend on these implementation details, this package uses an example goal-based user model and provides a parameterization example for the Fetch mobile manipulator's configuration details. Based on the paper "Assistance in Teleoperation of Redundant Robots through Predictive Joint Maneuvering".

# Code Organization

The primary PrediKCS algorithm search routine is implemented in `predikcs/search/` through the `VooBandit` class. This class makes use of utility classes stored in `predikcs/states/` for representing specific robot configurations, motion trajectories, and forward kinematics solvers. As PrediKCS also relies on a configurable user model, a base class `UserModel` and a goal-oriented `GoalClassifierUserModel` are provided as alternatives in `predikcs/user_models/`.

The PrediKCS algorithm takes as input:
+ A configuration for a redundant manipulator
+ The current set of joint positions
+ A task-space Twist command
+ A user model
+ Reward parameterization to perform a search over possible nullspace motion

In this package, `src/controller.cpp` collects and organizes this data from roslaunch parameters and incoming topics. The PrediKCS algorithm is run by calling the member function `GetCurrentBestJointVelocities(...)` on an object of type `VooBandit`. The control node here listens to joint position updates, updates the user model based on observed Twist commands, and repeatedly calls this search function. The resulting joint velocities are output at a fixed rate for execution by the manipulator.

Joint limit data specific to the Fetch manipulator has been stored as parameters in the launch file, while the node receives the configuration of the robot as a URDF published to the rosparam server. However, various tuning parameters present throughout the code may need to be altered for other manipulators as well (e.g. joint acceleration models used for integrating estimated motion during trajectory rollouts).

# Usage
This code is provided as a reference example purely for academic purposes. As it contains some sub-optimal implementation details, it is recommended that interested users implement the PrediKCS algorithm in their own codebase using this repository as a reference rather than directly depending on this package.

For further questions or discussions on using PrediKCS, contact Connor Brooks at connormbrooks@gmail.com.
