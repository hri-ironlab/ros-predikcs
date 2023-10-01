/*
Defines a VOO Bandit object for maintaining a set of configuration samples created over a rolling window time period. Manages creation of new samples
and deleting of expired samples.

Author: Connor Brooks
*/

#ifndef PREDIKCS_VOO_BANDIT_H
#define PREDIKCS_VOO_BANDIT_H

// includes

#include <vector>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <ros/ros.h>
#include <random>
#include <kdl/frames.hpp>

#include "predikcs/states/config_sample.h"

namespace predikcs
{

// forward declares
class MotionState;
class RewardCalculator;
class RobotModel;
class UserModel;

struct VooSpec
{
    // Probability of creating a new sample uniformly over the search space. With (1 - probability), sample is created in best Voronoi cell instead.
    double uniform_sample_prob = 0.5;
    // How many timesteps of control to maintain samples for
    int tau = 1;
    // Number of rollouts per sample to create Monte Carlo estimate of reward
    int sample_rollouts = 1;
    // How many user actions to evaluate per sample rollout
    int rollout_steps = 1;
    // Time between user actions in sample rollouts
    double delta_t = 1.0;
    // Discount factor between timesteps on rewards seen during sample rollout
    double gamma = 1.0;
    // How much time can be used to create samples and determine target velocity, in seconds
    double sampling_time_limit = 1.0;
    // Max number of samples to generate when looking for one within a Voronoi Cell before just taking closest one
    int max_voronoi_samples = 1;
    // Estimate of how much time will pass between calls to GenerateSamples
    double sampling_loop_rate = 1.0;
};

struct normal_random_variable
{
    normal_random_variable(const Eigen::MatrixXd& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](double x) { return dist(gen); });
    }
};

class VooBandit : public boost::enable_shared_from_this<VooBandit>
{
public:
    VooBandit(boost::shared_ptr<VooSpec> bandit_spec, boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<RewardCalculator> reward_calculator);
    ~VooBandit()
    {}

    void Initialize();

    void GenerateSamples(const boost::shared_ptr<MotionState> current_state, const boost::shared_ptr<UserModel> user_model);

    void GetCurrentBestJointVelocities(const boost::shared_ptr<MotionState> current_state, const std::vector<double>& current_ee_command, std::vector<double>& best_joint_velocities_out) const;

    void GetBaselineJointVelocities(const boost::shared_ptr<MotionState> current_state, const std::vector<double>& current_ee_command, std::vector<double>& best_joint_velocities_out) const;

    void GetCurrentBestConfig(std::vector<double>& best_config_out) const;

    boost::shared_ptr<VooSpec> bandit_spec_;
private:
    double UpdateBestConfig(const boost::shared_ptr<MotionState> start_state, const boost::shared_ptr<UserModel> user_model);
    void RandomSample(std::vector<double>& sample_target_out) const;
    void VoronoiSample(std::vector<double>& sample_target_out, const boost::shared_ptr<ConfigSample> best_config) const;

    bool initialized;
    bool best_expiring;
    int no_joints;
    int total_rollout_steps;
    std::vector< std::vector<boost::shared_ptr<ConfigSample>> > samples;
    int next_samples_timestep;
    Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> sample_gaussian_covariance;
    mutable std::default_random_engine random_generator;
    mutable std::vector<std::uniform_real_distribution<double>> uniform_joint_distributions;
    mutable std::uniform_real_distribution<double> sample_type_distribution;
    boost::shared_ptr<ConfigSample> best_config_;
    std::vector<boost::shared_ptr<ConfigSample>> best_timestep_configs_;
    mutable std::mutex robot_model_lock;

    boost::shared_ptr<RobotModel> robot_model_;
    boost::shared_ptr<RewardCalculator> reward_calculator_;
};

}

#endif  // PREDIKCS_VOO_BANDIT_H