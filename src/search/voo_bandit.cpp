// includes
#include "predikcs/search/voo_bandit.h"
#include "predikcs/states/motion_state.h"
#include "predikcs/search/reward_calculator.h"
#include "predikcs/states/robot_model.h"
#include "predikcs/user_models/user_model.h"
#include <math.h>
#include <chrono>

namespace predikcs
{

VooBandit::VooBandit(boost::shared_ptr<VooSpec> bandit_spec, boost::shared_ptr<RobotModel> robot_model, boost::shared_ptr<RewardCalculator> reward_calculator) : random_generator(std::chrono::system_clock::now().time_since_epoch().count())
{
    bandit_spec_ = bandit_spec;
    robot_model_ = robot_model;
    reward_calculator_ = reward_calculator;
    best_expiring = false;
    best_config_ = boost::shared_ptr<ConfigSample>(new ConfigSample(robot_model_));
}

void VooBandit::Initialize()
{
    // Setup random objects for creating new sample configurations
    no_joints = robot_model_->GetNumberOfJoints();
    sample_gaussian_covariance = Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic>::Identity(no_joints,no_joints);
    sample_type_distribution = std::uniform_real_distribution<double>(0.0, 1.0);
    for(int i = 0; i < no_joints; ++i)
    {
        double joint_lower_limit = robot_model_->GetJointPosDownLimit(i);
        if(joint_lower_limit == -std::numeric_limits<double>::infinity())
        {
            joint_lower_limit = -1 * M_PI;
        }
        double joint_upper_limit = robot_model_->GetJointPosUpLimit(i);
        if(joint_upper_limit == std::numeric_limits<double>::infinity())
        {
            joint_upper_limit = M_PI;
        }
        std::uniform_real_distribution<double> joint_distribution(joint_lower_limit, joint_upper_limit);
        uniform_joint_distributions.push_back(joint_distribution);
        // Make covariance equal to 1/10th of the width of each dimension
        sample_gaussian_covariance(i, i) = 0.1 * (joint_upper_limit - joint_lower_limit);
    }

    //Initialize data structures for storing sample configurations
    boost::shared_ptr<ConfigSample> null_sample(new ConfigSample(robot_model_));
    best_timestep_configs_.clear();
    samples.clear();
    for(int i = 0; i < bandit_spec_->tau; ++i)
    {
        std::vector<boost::shared_ptr<ConfigSample>> timestep_samples;
        samples.push_back(timestep_samples);
        best_timestep_configs_.push_back(null_sample);
    }
    next_samples_timestep = -1;
    // Determine how many timesteps to generate per rollout given evaluation horizon, how long samples are kept, and sampling loop rate
    total_rollout_steps = ceil( ((bandit_spec_->tau * bandit_spec_->sampling_loop_rate) + (bandit_spec_->rollout_steps * bandit_spec_->delta_t)) / bandit_spec_->delta_t );
    initialized = true;
}

void VooBandit::GenerateSamples(const boost::shared_ptr<MotionState> current_state, const boost::shared_ptr<UserModel> user_model)
{
    // Delete old samples and find current best
    if(!initialized)
    {
        Initialize();
    }

    next_samples_timestep = (next_samples_timestep + 1) % samples.size();
    auto start_time = std::chrono::high_resolution_clock::now();
    robot_model_lock.lock();
    double best_config_reward = UpdateBestConfig(current_state, user_model);
    robot_model_lock.unlock();
    samples[next_samples_timestep].clear();
    bool time_remaining = true;
    while(time_remaining)
    {
        auto sample_start_time = std::chrono::high_resolution_clock::now();
        std::vector<double> sample_target;
        if(best_expiring)
        {
            for(int i = 0; i < best_config_->target_joint_pos.size(); ++i)
            {
                sample_target.push_back(best_config_->target_joint_pos[i]);
            }
            best_expiring = false;
        }
        else if(best_config_reward == -std::numeric_limits<double>::infinity() || sample_type_distribution(random_generator) < bandit_spec_->uniform_sample_prob)
        {
            RandomSample(sample_target);
        }
        else
        {
            VoronoiSample(sample_target, best_config_);
        }
        boost::shared_ptr<ConfigSample> new_sample( new ConfigSample(sample_target, robot_model_, reward_calculator_));
        robot_model_lock.lock();
        new_sample->GenerateRollouts(current_state, bandit_spec_->sample_rollouts, total_rollout_steps, bandit_spec_->delta_t, user_model);
        robot_model_lock.unlock();
        double new_sample_reward = new_sample->GetExpectedReward(0, bandit_spec_->rollout_steps, bandit_spec_->gamma, current_state->joint_positions, user_model);
        if(new_sample_reward > best_config_reward)
        {
            best_config_ = new_sample;
            best_config_reward = new_sample_reward;
        }
        samples[next_samples_timestep].push_back(new_sample);

        auto sample_stop = std::chrono::high_resolution_clock::now();
        double sample_duration = std::chrono::duration<double>(sample_stop - sample_start_time).count();
        double algo_duration = std::chrono::duration<double>(sample_stop - start_time).count();
        double next_sample_total_duration_secs = algo_duration + sample_duration;
        if(next_sample_total_duration_secs > bandit_spec_->sampling_time_limit)
        {
            time_remaining = false;
        }
    }
}

void VooBandit::GetCurrentBestJointVelocities(const boost::shared_ptr<MotionState> current_state, const std::vector<double>& current_ee_command, std::vector<double>& best_joint_velocities_out) const
{
    robot_model_lock.lock();
    best_config_->GetJointVelocities(current_state, current_ee_command, best_joint_velocities_out);
    robot_model_lock.unlock();
}

void VooBandit::GetBaselineJointVelocities(const boost::shared_ptr<MotionState> current_state, const std::vector<double>& current_ee_command, std::vector<double>& best_joint_velocities_out) const
{
    robot_model_lock.lock();
    ConfigSample baseline_config(robot_model_);
    baseline_config.GetJointVelocities(current_state, current_ee_command, best_joint_velocities_out);
    robot_model_lock.unlock();
}

void VooBandit::GetCurrentBestConfig(std::vector<double>& best_config_out) const
{
    best_config_out.clear();
    for(int i = 0; i < best_config_->target_joint_pos.size(); ++i)
    {
        best_config_out.push_back(best_config_->target_joint_pos[i]);
    }
}

// -------------------- Private Methods ------------------------------

double VooBandit::UpdateBestConfig(boost::shared_ptr<MotionState> start_state, boost::shared_ptr<UserModel> user_model)
{
    // Re-calculate best reward given the current timestep starting point
    best_config_ = boost::shared_ptr<ConfigSample>(new ConfigSample(robot_model_));
    double best_config_reward = -std::numeric_limits<double>::infinity();
    best_expiring = false;
    double timestep = 0;
    double total_counted = 0;
    for(int i = 0; i < samples.size(); ++i)
    {
        // Calculate which step of rollout to start reward calculation from based on how old this sample is
        double time_passed = 0.0;
        bool expir_flag = false;
        if(i == next_samples_timestep)
        {
            time_passed = (samples.size() - 1) * bandit_spec_->sampling_loop_rate;
            expir_flag = true;
        }
        else if(next_samples_timestep > i)
        {
            time_passed = (next_samples_timestep - i) * bandit_spec_->sampling_loop_rate;
        }
        else
        {
            time_passed = (next_samples_timestep + (samples.size() - i)) * bandit_spec_->sampling_loop_rate;
        }
        int starting_timestep = round( time_passed / bandit_spec_->delta_t );

        double counted = 0;
        std::vector<boost::shared_ptr<ConfigSample>>::iterator it;
        for(it = samples[i].begin(); it != samples[i].end();)
        {
            double sample_reward = (*it)->GetExpectedReward(starting_timestep, bandit_spec_->rollout_steps, bandit_spec_->gamma, start_state->joint_positions, user_model);
            if(sample_reward == -std::numeric_limits<double>::infinity())
            {
                // Remove old sample that has gone bad on distance metric
                it = samples[i].erase(it);
                continue;
            }

            if(best_config_reward == -std::numeric_limits<double>::infinity() || sample_reward > best_config_reward)
            {
                best_config_ = *it;
                best_config_reward = sample_reward;
                if(expir_flag)
                {
                    best_expiring = true;
                }
                else
                {
                    best_expiring = false;
                }
            }
            ++counted;
            ++it;
        }
        if(samples[i].size() > 0)
        {
            //ROS_ERROR("%d old, %d left", starting_timestep, samples[i].size());
            total_counted += counted;
        }   
    }
    ROS_ERROR("Total samples considered: %.1f", total_counted);
    return best_config_reward;
}

void VooBandit::RandomSample(std::vector<double>& sample_target_out) const
{
    // Create new random sample
    for(int i = 0; i < no_joints; ++i)
    {
        sample_target_out.push_back(uniform_joint_distributions[i](random_generator));
    }
}
void VooBandit::VoronoiSample(std::vector<double>& sample_target_out, const boost::shared_ptr<ConfigSample> best_config) const
{
    // Create new sample in best Voronoi cell
    Eigen::VectorXd voronoi_cell_mean(no_joints);
    for(int i = 0; i < no_joints; ++i)
    {
        voronoi_cell_mean(i) = best_config->target_joint_pos[i];
    }
    normal_random_variable noise_sample { voronoi_cell_mean, sample_gaussian_covariance };
    Eigen::VectorXd new_sample(no_joints);
    Eigen::VectorXd best_sample(no_joints);
    double best_sample_dist = -1.0;
    // Create up to max number of samples and check if each one is within this Voronoi cell. Otherwise, take closest one
    for(int i = 0; i < bandit_spec_->max_voronoi_samples; ++i)
    {
        new_sample = noise_sample();
        double dist_to_center = best_config->GetDistToSample(new_sample);
        // Check if within Voronoi Cell
        bool in_voronoi = true;
        for(int j = 0; j < samples.size(); ++j)
        {
            for(int k = 0; k < samples[j].size(); ++k)
            {
                double dist_to_sample = samples[j][k]->GetDistToSample(new_sample);
                if(dist_to_sample < dist_to_center)
                {
                    in_voronoi = false;
                    break;
                }
            }
            if(!in_voronoi)
            {
                break;
            }
        }
        if(in_voronoi)
        {
            best_sample = new_sample;
            break;
        }
        else if(best_sample_dist == -1.0 || dist_to_center < best_sample_dist)
        {
            best_sample = new_sample;
            best_sample_dist = dist_to_center;
        }
    }
    for(int i = 0; i < no_joints; ++i)
    {
        sample_target_out.push_back(best_sample(i));
    }
    
}

}