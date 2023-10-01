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

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

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

inline double GetLinearDistance(const KDL::Vector& position_1, const KDL::Vector& position_2);

inline double GetAngularDistance(const KDL::Rotation& rotation_1, const KDL::Rotation& rotation_2);

inline double CalculateDistance(const KDL::Frame& frame_1, const KDL::Frame& frame_2, const double linear_distance_weight, const double angular_distance_weight);

inline double CalculateSmoothness(const std::vector<double>& old_accels, const std::vector<double>& new_accels, const double timestep);

inline double CalculateManipulability(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> candidate_motion);

inline double CalculateLimitCloseness(const boost::shared_ptr<RobotModel> robot_model, const boost::shared_ptr<MotionState> candidate_motion);

}

#endif  // PREDIKCS_REWARD_CALCULATOR_H