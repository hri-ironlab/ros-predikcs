/*
Defines a particular target joint configuration for use by a VOO Bandit object.
Author: Connor Brooks
*/

#ifndef PREDIKCS_CONFIG_SAMPLE_H
#define PREDIKCS_CONFIG_SAMPLE_H

// includes

#include <vector>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <ros/ros.h>
#include <random>
#include <mutex>
#include <kdl/frames.hpp>

namespace predikcs
{

// forward declares
class MotionState;
class RewardCalculator;
class RobotModel;
class UserModel;

class ConfigSample : public boost::enable_shared_from_this<ConfigSample>
{
public:
    ConfigSample(boost::shared_ptr<RobotModel> robot_model);
    ConfigSample(std::vector<double> target_joint_positions, boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<RewardCalculator> reward_calculator);
    ~ConfigSample()
    {}

    void ResetSample();

    double GetExpectedReward(const int start_timestep, const int num_timesteps, const double discount_factor, const std::vector<double>& joint_pos, const boost::shared_ptr<UserModel> user_model, const double joint_dist_cutoff = 0.5) const;

    void GenerateRollouts(const boost::shared_ptr<MotionState> starting_state, const int num_rollouts, const int num_timesteps, const double timestep_size, const boost::shared_ptr<UserModel> user_model);

    void GenerateRollout(const boost::shared_ptr<MotionState> starting_state, const int num_timesteps, const double timestep_size, const boost::shared_ptr<UserModel> user_model, std::vector<std::vector<double>>& predicted_joint_states, std::vector<double>& rollout_rewards, std::pair<int, double>& sample_probs);

    double GetDistToSample(const Eigen::VectorXd& sample) const;

    void GetJointVelocities(const boost::shared_ptr<MotionState> joint_start_state, std::vector<double> ee_command, std::vector<double>& joint_vels_out, const double gain = 0.2) const;

    std::vector<double> target_joint_pos;
private:
    bool samples_generated;
    bool null_sample;
    std::mutex thread_mod_lock;
    std::vector<std::vector<std::vector<double>>> predicted_joint_states_per_roll;
    std::vector<std::vector<double>> avg_timestep_scores_per_roll;
    std::vector<std::pair<int, double>> rollout_sample_probabilities;
    boost::shared_ptr<RobotModel> robot_model_;
    boost::shared_ptr<RewardCalculator> reward_calculator_;
};

}

#endif  // PREDIKCS_CONFIG_SAMPLE_H