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

class VooSpec
{
public:
    VooSpec()
    {}
    ~VooSpec()
    {}
    // Probability of creating a new sample uniformly over the search space. With (1 - probability), sample is created in best Voronoi cell instead.
    double uniform_sample_prob;
    // How many timesteps of control to maintain samples for
    int tau;
    // Number of rollouts per sample to create Monte Carlo estimate of reward
    int sample_rollouts;
    // How many user actions to evaluate per sample rollout
    int rollout_steps;
    // Time between user actions in sample rollouts
    double delta_t;
    // Discount factor between timesteps on rewards seen during sample rollout
    double gamma;
    // How much time can be used to create samples and determine target velocity, in seconds
    double sampling_time_limit;
    // Max number of samples to generate when looking for one within a Voronoi Cell before just taking closest one
    int max_voronoi_samples;
    // Estimate of how much time will pass between calls to GenerateSamples
    double sampling_loop_rate;
};

struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
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

    void GenerateSamples(boost::shared_ptr<MotionState> current_state, boost::shared_ptr<UserModel> user_model);

    void GetCurrentBestJointVelocities(boost::shared_ptr<MotionState> current_state, std::vector<double>* current_ee_command, std::vector<double>* best_joint_velocities);

    void GetBaselineJointVelocities(boost::shared_ptr<MotionState> current_state, std::vector<double>* current_ee_command, std::vector<double>* best_joint_velocities);

    void GetCurrentBestConfig(std::vector<double>* best_config);

    boost::shared_ptr<VooSpec> bandit_spec_;
private:
    double UpdateBestConfig(boost::shared_ptr<MotionState> start_state, boost::shared_ptr<UserModel> user_model);
    void RandomSample(std::vector<double> &sample_target);
    void VoronoiSample(std::vector<double> &sample_target, boost::shared_ptr<ConfigSample> best_config);

    bool initialized;
    bool best_expiring;
    int no_joints;
    int total_rollout_steps;
    std::vector< std::vector<boost::shared_ptr<ConfigSample>> > samples;
    int next_samples_timestep;
    Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> sample_gaussian_covariance;
    std::default_random_engine random_generator;
    std::vector<std::uniform_real_distribution<double>> uniform_joint_distributions;
    std::uniform_real_distribution<double> sample_type_distribution;
    boost::shared_ptr<ConfigSample> best_config_;
    std::vector<boost::shared_ptr<ConfigSample>> best_timestep_configs_;
    std::mutex robot_model_lock;

    boost::shared_ptr<RobotModel> robot_model_;
    boost::shared_ptr<RewardCalculator> reward_calculator_;
};

}

#endif  // PREDIKCS_VOO_BANDIT_H