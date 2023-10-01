/*
Defines a user model class for use in Iron Lab's Predictive Velocity Controller.

Author: Connor Brooks
*/

#ifndef PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H
#define PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H

//includes
#include <vector>
#include <mutex>
#include <utility>
#include <random>
#include <boost/shared_ptr.hpp>
#include <kdl/frames.hpp>
#include "predikcs/user_models/user_model.h"
#include "predikcs/search/reward_calculator.h"

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

class GoalClassifierUserModel : public UserModel
{
public:

    GoalClassifierUserModel(const int num_options, const double action_timestep);

    std::pair<int, double> RandomSample(const boost::shared_ptr<MotionState> state, std::vector<double>& sample, const int sample_bias) const override;

    double GetSampleProbability(const int sample_bias) const override;

    void SetRobotModel(boost::shared_ptr<RobotModel> robot_model) { robot_model_ = robot_model; }

    void SetGoals(const std::vector<std::vector<double>>& goal_points);

    void GetProbabilities(std::vector<double>& probabilities) const;

    void ResetProbabilities();

    void UpdateProbabilities(const boost::shared_ptr<MotionState> state, const std::vector<double>& velocities);

private:
    void CalculateProbabilities() const;

    boost::shared_ptr<RobotModel> robot_model_;
    std::vector<KDL::Frame> goals;
    std::vector<double> log_likelihoods;

    // Cache variables
    mutable bool probs_updated_since_last_calc;
    mutable std::mutex goal_prob_lock;
    mutable std::vector<double> goal_probabilities;
};

}

#endif  // PREDIKCS_GOAL_CLASSIFIER_USER_MODEL_H