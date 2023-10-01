// includes
#include "predikcs/states/config_sample.h"
#include "predikcs/states/motion_state.h"
#include "predikcs/search/reward_calculator.h"
#include "predikcs/states/robot_model.h"
#include "predikcs/user_models/user_model.h"
#include <math.h>
#include <chrono>
#include <thread>

namespace predikcs
{

double GetJointDist(const boost::shared_ptr<RobotModel> robot_model_, const std::vector<double>& joint_pos_1, const std::vector<double>& joint_pos_2)
{
    double joint_dist_sum = 0.0;
    for(int i = 0; i < joint_pos_1.size(); ++i)
    {
        double joint_dist = abs(joint_pos_1[i] - joint_pos_2[i]);
        if(joint_dist > M_PI && robot_model_->GetJointPosUpLimit(i) == std::numeric_limits<double>::infinity())
        {
            joint_dist = abs(joint_dist - (2 * M_PI));
        }
        joint_dist_sum += pow(joint_dist, 2);
    }
    return pow(joint_dist_sum, 0.5);
}

ConfigSample::ConfigSample(boost::shared_ptr<RobotModel> robot_model) : robot_model_(robot_model)
{
    // Null sample, has negative infinity reward
    null_sample = true;
}

ConfigSample::ConfigSample(std::vector<double> target_joint_positions, boost::shared_ptr<RobotModel> robot_model,
                           boost::shared_ptr<RewardCalculator> reward_calculator) : target_joint_pos(target_joint_positions), robot_model_(robot_model), reward_calculator_(reward_calculator)
{
    samples_generated = false;
    null_sample = false;
}

void ConfigSample::ResetSample()
{
    samples_generated = false;
    predicted_joint_states_per_roll.clear();
    avg_timestep_scores_per_roll.clear();
}

double ConfigSample::GetExpectedReward(const int start_timestep, const int num_timesteps, const double discount_factor, const std::vector<double>& joint_pos, const boost::shared_ptr<UserModel> user_model, const double joint_dist_cutoff) const
{
    if(null_sample)
    {
        return -std::numeric_limits<double>::infinity();
    }

    if(!samples_generated || num_timesteps <= 0 || start_timestep < 0 || avg_timestep_scores_per_roll.size() < 1 || start_timestep + num_timesteps > avg_timestep_scores_per_roll[0].size())
    {
        return -std::numeric_limits<double>::infinity();
    }

    int num_rollouts_counted = 0;
    double corrected_avg_reward = 0.0;
    for(int roll = 0; roll < avg_timestep_scores_per_roll.size(); ++roll)
    {
        double joint_dist_sum = GetJointDist(robot_model_, joint_pos, predicted_joint_states_per_roll[roll][start_timestep]);
        int roll_start_timestep = start_timestep;
        if(joint_dist_sum > joint_dist_cutoff)
        {
            // Try before and after timesteps
            double pre_joint_dist_sum = std::numeric_limits<double>::infinity();
            double post_joint_dist_sum = std::numeric_limits<double>::infinity();
            if(start_timestep > 0)
            {
                pre_joint_dist_sum = GetJointDist(robot_model_, joint_pos, predicted_joint_states_per_roll[roll][start_timestep - 1]);
            }
            if(start_timestep < predicted_joint_states_per_roll[roll].size() - 1)
            {
                post_joint_dist_sum = GetJointDist(robot_model_, joint_pos, predicted_joint_states_per_roll[roll][start_timestep + 1]);
            }
            if(pre_joint_dist_sum > joint_dist_cutoff && post_joint_dist_sum > joint_dist_cutoff)
            {
                continue;
            }
            else if (pre_joint_dist_sum < post_joint_dist_sum)
            {
                roll_start_timestep = start_timestep - 1;
            }
            else
            {
                roll_start_timestep = start_timestep + 1;
            }
        }

        num_rollouts_counted++;
        double discount = 1.0;
        double combined_reward = 0.0;
        for(int i = roll_start_timestep; i < roll_start_timestep + num_timesteps; ++i)
        {
            combined_reward += discount * avg_timestep_scores_per_roll[roll][i];
            discount = discount * discount_factor;
        }
        double new_prob = user_model->GetSampleProbability(rollout_sample_probabilities[roll].first);
        double importance_corrected_reward = combined_reward * (new_prob / rollout_sample_probabilities[roll].second);
        corrected_avg_reward += importance_corrected_reward;
    }

    if(num_rollouts_counted == 0)
    {
        return -std::numeric_limits<double>::infinity();
    }
    else
    {
        return corrected_avg_reward / num_rollouts_counted;
    }
}

void ConfigSample::GenerateRollouts(const boost::shared_ptr<MotionState> starting_state, const int num_rollouts, const int num_timesteps, const double timestep_size, const boost::shared_ptr<UserModel> user_model)
{
    avg_timestep_scores_per_roll.clear();
    avg_timestep_scores_per_roll.resize(num_rollouts);
    predicted_joint_states_per_roll.clear();
    predicted_joint_states_per_roll.resize(num_rollouts);
    rollout_sample_probabilities.clear();
    rollout_sample_probabilities.resize(num_rollouts);

    std::vector<std::thread> threads;
    for(int i = 0; i < num_rollouts; ++i)
    {
        threads.push_back(std::thread(&ConfigSample::GenerateRollout, this, starting_state, num_timesteps, timestep_size, user_model, std::ref(predicted_joint_states_per_roll[i]), std::ref(avg_timestep_scores_per_roll[i]), std::ref(rollout_sample_probabilities[i])));
    }

    for(int i = 0; i < num_rollouts; ++i)
    {
        threads[i].join();
    }
    samples_generated = true;
}

void ConfigSample::GenerateRollout(const boost::shared_ptr<MotionState> starting_state, const int num_timesteps, const double timestep_size, const boost::shared_ptr<UserModel> user_model, std::vector<std::vector<double>>& predicted_joint_states_out, std::vector<double>& rollout_rewards_out, std::pair<int, double>& sample_probs_out)
{
    std::vector<boost::shared_ptr<MotionState>> rollout_states;
    rollout_states.push_back(starting_state);
    double time_into_future = starting_state->time_in_future;
    std::pair<int, double> user_model_sample(-1, 0.0);
    for(int j = 0; j < num_timesteps; ++j)
    {
        time_into_future += timestep_size;

        // Sample new user command
        std::vector<double> next_command;
        std::vector<double> next_desired_movement;
        user_model_sample = user_model->RandomSample(rollout_states[rollout_states.size() - 1], next_command, user_model_sample.first);
        for(int k = 0; k < next_command.size(); ++k)
        {
            next_desired_movement.push_back(next_command[k] * timestep_size);
        }
        
        // Calculate new joint velocities motion candidate
        std::vector<double> next_joint_vels;
        GetJointVelocities(rollout_states[rollout_states.size() - 1], next_command, next_joint_vels);
        
        // Update state
        KDL::Frame ideal_position( KDL::Rotation::RPY(next_desired_movement[3], next_desired_movement[4], next_desired_movement[5]) * rollout_states[rollout_states.size() - 1]->position.M, 
            rollout_states[rollout_states.size() - 1]->position.p + KDL::Vector(next_desired_movement[0], next_desired_movement[1], next_desired_movement[2]));
        boost::shared_ptr<MotionState> new_state(new MotionState(rollout_states[rollout_states.size() - 1]->joint_positions, rollout_states[rollout_states.size() - 1]->joint_velocities, 
            next_joint_vels, timestep_size, robot_model_, time_into_future));
        // Calculate reward for new motion candidate
        double timestep_score = reward_calculator_->EvaluateMotionCandidate(robot_model_, rollout_states[rollout_states.size() - 1], new_state, ideal_position, false);
        thread_mod_lock.lock();
        rollout_rewards_out.push_back(timestep_score);
        predicted_joint_states_out.push_back(rollout_states[rollout_states.size() - 1]->joint_positions);
        thread_mod_lock.unlock();
        rollout_states.push_back(new_state);
    }
    sample_probs_out.first = user_model_sample.first;
    sample_probs_out.second = user_model_sample.second;
}

double ConfigSample::GetDistToSample(const Eigen::VectorXd& sample) const
{
    double sq_sum = 0.0;
    for(int i = 0; i < target_joint_pos.size(); ++i)
    {
        sq_sum += pow(target_joint_pos[i] - sample(i), 2.0);
    }
    return pow(sq_sum, 0.5);
}

void ConfigSample::GetJointVelocities(const boost::shared_ptr<MotionState> joint_start_state, std::vector<double> ee_command, std::vector<double>& joint_vels_out, const double gain) const
{
    // Calculate primary movement
    const Eigen::VectorXd desired_velocity = Eigen::Map<Eigen::VectorXd>(&(ee_command[0]), ee_command.size());
    joint_start_state->CalculateJacobian(robot_model_);
    Eigen::VectorXd primary_movement = joint_start_state->pseudo_inverse * desired_velocity;

    // Calculate null space matrix
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> ident = Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic>::Identity(joint_start_state->pseudo_inverse.rows(),joint_start_state->jacobian.cols());
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> null_space_ = ident - joint_start_state->pseudo_inverse * joint_start_state->jacobian;

    // Determine null movement goal vector based on difference between current and target joint positions
    Eigen::VectorXd null_movement_eigen(robot_model_->GetNumberOfJoints());
    for(int i = 0; i < robot_model_->GetNumberOfJoints(); ++i)
    {
        if(null_sample)
        {
            break;
        }
        
        null_movement_eigen(i) = target_joint_pos[i] - joint_start_state->joint_positions[i];
        if(abs(null_movement_eigen(i)) > M_PI && robot_model_->GetJointPosUpLimit(i) == std::numeric_limits<double>::infinity())
        {
            if(null_movement_eigen(i) < 0)
            {
                null_movement_eigen(i) = (2 * M_PI) + null_movement_eigen(i);
            }
            else
            {
                null_movement_eigen(i) = -((2 * M_PI) - null_movement_eigen(i));
            }
        }
        null_movement_eigen(i) = gain * null_movement_eigen(i);
    }

    // Calculate total movement and store in given vector
    Eigen::VectorXd total_movement = primary_movement;
    if(!null_sample)
    {
        total_movement = total_movement + null_space_*null_movement_eigen;
    }
    joint_vels_out.clear();
    for(int i = 0; i < robot_model_->GetNumberOfJoints(); ++i)
    {
        if(abs(total_movement(i)) > robot_model_->GetJointVelLimit(i))
        {
            if(total_movement(i) > 0)
            {
                total_movement(i) = robot_model_->GetJointVelLimit(i);
            }
            else
            {
                total_movement(i) = -1 * robot_model_->GetJointVelLimit(i);
            }
        }
        joint_vels_out.push_back(total_movement(i));
    }
}

}