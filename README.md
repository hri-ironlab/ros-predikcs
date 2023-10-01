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

In this package, `src/controller.cpp` collects and organizes this data from roslaunch parameters and incoming topics. The PrediKCS algorithm is run by calling `GetSamples(...)` on a `VooBandit` object to generate samples, followed by passing a Twist command for translation into the `GetCurrentBestJointVelocities(...)` function on the same object. The control node here listens to joint position updates, updates the user model based on observed Twist commands, and calls this translation function before outputting the resulting joint velocities for execution by the manipulator. Then, the command node runs the sampling routine for the rest of the control loop's update time.

Joint limit data specific to the Fetch manipulator has been stored as parameters in the launch file. The node receives the configuration of the robot as a URDF published to the rosparam server, the name for which is also provided as a launch file parameter. However, various tuning parameters present throughout the code may need to be altered for use with other manipulators (e.g. joint acceleration models used for integrating estimated motion during trajectory rollouts).

A note on implementation: the control loop is kept single-threaded and executes at a fixed rate in this package for the sake of comparision with other controllers. An alternative implementation requiring minimal changes is to run these routines in separate threads. Since the `GetCurrentBestJointVelocities(...)` function only requires a retrieval of the latest best nullspace target configuration and a calculation of the current Jacobian, this alternative multi-threaded approach would allow continual sampling with event-based joint velocity outputs anytime new Twist commands are received.

# Usage
This code is provided as a reference example purely for academic purposes. As it contains some sub-optimal implementation details, it is recommended that interested users implement the PrediKCS algorithm in their own codebase using this repository as a reference rather than directly depending on this package.

For further questions or discussions on using PrediKCS, contact Connor Brooks at connormbrooks@gmail.com.
