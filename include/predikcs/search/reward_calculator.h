/*
Defines a base class for reward calculators for use in Iron Lab's Predictive Velocity Controller.
Subclasses from this base class should implement specific types of reward calculations.

Author: Connor Brooks
*/

#ifndef PREDIKCS_REWARD_CALCULATOR_H
#define PREDIKCS_REWARD_CALCULATOR_H

//includes
#include <vector>
#include <kdl/frames.hpp>
#include <Eigen/LU>
#include <boost/shared_ptr.hpp>
#include "predikcs/states/robot_model.h"
#include "predikcs/states/motion_state.h"

namespace predikcs
{

class RewardCalculator
{
public:
    RewardCalculator() = default;

    double EvaluateMotionCandidate(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> old_state, const boost::shared_ptr<MotionState> candidate_motion, const KDL::Frame& ideal_position, bool verbose) const;

    void SetParameters(const double distance_weight, const double jerk_weighting, const double manipulability_weight, const double limits_weight);

    double GetDistWeight() const { return dist_weight; }
    double GetJerkWeight() const { return jerk_weight; }
    double GetManipWeight() const { return manip_weight; }
    double GetLimWeight() const { return lim_weight; }

private:
    double dist_weight = -1.0;
    double jerk_weight = -1.0;
    double manip_weight = -1.0;
    double lim_weight = -1.0;
};

//Euclidean distance between two KDL Vectors
inline double GetLinearDistance(const KDL::Vector& position_1, const KDL::Vector& position_2)
{
    return sqrt(pow((position_1.data[0] - position_2.data[0]),2) + pow((position_1.data[1] - position_2.data[1]),2) 
        + pow((position_1.data[2] - position_2.data[2]),2));
}

//Distance between two KDL Rotations using 2 times arccos of dot product of the two quaternions
inline double GetAngularDistance(const KDL::Rotation& rotation_1, const KDL::Rotation& rotation_2)
{
    double rot1_x, rot1_y, rot1_z, rot1_w;
    rotation_1.GetQuaternion(rot1_x, rot1_y, rot1_z, rot1_w);
    double rot2_x, rot2_y, rot2_z, rot2_w;
    rotation_2.GetQuaternion(rot2_x, rot2_y, rot2_z, rot2_w);
    double dot_product = std::min(abs(rot1_x*rot2_x + rot1_y*rot2_y + rot1_z*rot2_z + rot1_w*rot2_w), 1.0);
    return 2*acos(dot_product);
}

inline double CalculateDistance(const KDL::Frame& frame_1, const KDL::Frame& frame_2, const double linear_distance_weight, const double angular_distance_weight)
{
    double linear_distance = GetLinearDistance(frame_1.p, frame_2.p);
    double angular_distance = GetAngularDistance(frame_1.M, frame_2.M);
    return pow(linear_distance, 2)*linear_distance_weight + pow(angular_distance, 2)*angular_distance_weight;
}

// Calculates smoothness as the L2 norm of the jerk.
inline double CalculateSmoothness(const std::vector<double>& old_accel, const std::vector<double>& new_accel, const double timestep)
{
    double sum_squared_diffs = 0.0;
    for(int i = 0; i < new_accel.size(); i++)
    {
        sum_squared_diffs += pow((new_accel[i] - old_accel[i]) / timestep,2);
    }

    return sqrt(sum_squared_diffs);
}

// Retrieves the Yoshikawa manipulability measure for singularity avoidance/closeness to singularities
inline double CalculateManipulability(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> candidate_motion) {
    candidate_motion->CalculateManipulability(robot_model);
    return candidate_motion->manipulability;
}

// Returns the number of joints sufficiently close to their limit
inline double CalculateLimitCloseness(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> candidate_motion, const double limit_closeness = 0.1) {
    double num_limit_joints = 0.0;
    for(int i = 0; i < candidate_motion->joint_positions.size(); ++i)
    {
        if(robot_model->GetJointPosUpLimit(i) != std::numeric_limits<double>::infinity()){
            double min_dist = std::min(candidate_motion->joint_positions[i] - robot_model->GetJointPosDownLimit(i), robot_model->GetJointPosUpLimit(i) - candidate_motion->joint_positions[i]);
            if(min_dist < limit_closeness)
            {
                num_limit_joints += 1.0;
            }
        }
    }
    return num_limit_joints;
}

}

#endif  // PREDIKCS_REWARD_CALCULATOR_H