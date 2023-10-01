/*
Most basic version of a reward calculator. Rewards smooth motion that properly applies velocity commands.
Subclasses of this class can be used for alternative versions of reward calculator.

Author: Connor Brooks
*/

//includes
#include "predikcs/search/reward_calculator.h"
#include "predikcs/states/motion_state.h"
#include <cmath>

namespace predikcs
{

// Master function for determining the reward a possible motion candidate gets
// This version includes a weighted sum of:
// 1. Distance between "idealized" motion result and this motion result (using assumption robot speed: radians/sec ~= 4*meters/sec)
// 2, Motion smoothness
// 3. Closeness to singularities
// 4. Closeness to joint limits
double RewardCalculator::EvaluateMotionCandidate(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> old_state, const boost::shared_ptr<MotionState> candidate_motion, const KDL::Frame& ideal_position, const bool verbose) const
{
    candidate_motion->CalculatePosition(robot_model);

    // Calculate distance between resulting position and position from idealized movement
    double x, y, z, w;
    candidate_motion->position.M.GetQuaternion(x, y, z, w);

    if(verbose)
    {
        ROS_DEBUG("Motion Candidate resulting pose: %.2f %.2f %.2f %.2f %.2f %.2f %.2f", candidate_motion->position.p.x(), candidate_motion->position.p.y(), candidate_motion->position.p.z(), x, y, z, w);
        ROS_DEBUG("Motion Candidate commanded velocities: %.2f %.2f %.2f %.2f %.2f %.2f %.2f", candidate_motion->commanded_velocities[0], candidate_motion->commanded_velocities[1], candidate_motion->commanded_velocities[2], candidate_motion->commanded_velocities[3], candidate_motion->commanded_velocities[4], candidate_motion->commanded_velocities[5], candidate_motion->commanded_velocities[6]);
        ROS_DEBUG("Motion Candidate resulting velocities: %.2f %.2f %.2f %.2f %.2f %.2f %.2f", candidate_motion->joint_velocities[0], candidate_motion->joint_velocities[1], candidate_motion->joint_velocities[2], candidate_motion->joint_velocities[3], candidate_motion->joint_velocities[4], candidate_motion->joint_velocities[5], candidate_motion->joint_velocities[6]);
    }
    
    double distance_estimate = CalculateDistance(candidate_motion->position, ideal_position, 1.0, 1.0);

    // Calculate motion smoothness as determined by size of jerk for each joint
    double jerk_size = CalculateSmoothness(old_state->joint_accelerations, candidate_motion->joint_accelerations, candidate_motion->time_in_future - old_state->time_in_future);

    // Estimate closeness to singularities
    double manipulability = 0.0;
    if(manip_weight != 0.0)
    {
        candidate_motion->CalculateJacobian(robot_model);
        manipulability = CalculateManipulability(robot_model, candidate_motion);
    }

    double num_limited_joints = CalculateLimitCloseness(robot_model, candidate_motion);

    if(verbose)
    {
        ROS_DEBUG("Distance Score: %.2f", distance_estimate);
        ROS_DEBUG("jerk size: %.2f", jerk_size);
        ROS_DEBUG("manipulability: %.2f", manipulability);
        ROS_DEBUG("limits: %.2f", num_limited_joints);
        ROS_DEBUG("individual weighted components: %.3f, %.3f, %.3f, %.3f", dist_weight*distance_estimate, jerk_weight*jerk_size, manip_weight*manipulability, lim_weight * num_limited_joints);
    }

    return (dist_weight * distance_estimate) + (jerk_weight * jerk_size) + (manip_weight * manipulability) + (lim_weight * num_limited_joints);
}

void RewardCalculator::SetParameters(double distance_weight, double jerk_weighting, double manipulability_weight, double limits_weight)
{
    dist_weight = distance_weight;
    jerk_weight = jerk_weighting;
    manip_weight = manipulability_weight;
    lim_weight = limits_weight;
}

}