//User model that just assigns probability based on distance to given waypoint pose

//includes
#include "predikcs/user_models/goal_classifier_user_model.h"
#include "predikcs/states/motion_state.h"
#include <cmath>
#include <ros/console.h>
#include <chrono>
#include <algorithm>

namespace predikcs
{

//----------------------------------------------------------------------------------------------------------------------------
// GoalClassifierUserModel definition

GoalClassifierUserModel::GoalClassifierUserModel(const int num_options, const double action_timestep) : UserModel(num_options, action_timestep)
{ 
    SetNumOptions(num_options);
    SetActionTimestep(action_timestep);
    for(int i = 0; i < 6; ++i)
    {
        last_velocity_command_.push_back(0.0);
    }
    probs_updated_since_last_calc = true;
}

void GoalClassifierUserModel::SetGoals(const std::vector<std::vector<double>>& goal_points)
{
    goals.clear();
    for(int i = 0; i < goal_points.size(); ++i)
    {
        KDL::Frame new_goal( 
            KDL::Rotation::Quaternion(goal_points[i][3], goal_points[i][4], goal_points[i][5], goal_points[i][6]),
            KDL::Vector(goal_points[i][0], goal_points[i][1], goal_points[i][2]));
        goals.push_back(new_goal);
    }
    ResetProbabilities();
}

void GoalClassifierUserModel::ResetProbabilities()
{
    log_likelihoods.clear();
    for(int i = 0; i < goals.size(); ++i)
    {
        log_likelihoods.push_back(0.0);
    }
    probs_updated_since_last_calc = true;
}

void GoalClassifierUserModel::UpdateProbabilities(const boost::shared_ptr<MotionState> state, const std::vector<double>& velocities)
{
    // Calculate reward (improvement in distance) for each goal
    state->CalculatePosition(robot_model_);
    double reward_sum = 0.0;
    double dist_factor = 1.0;
    std::vector<double> rewards;
    for(int i = 0; i < goals.size(); ++i)
    {
        double dist_to_start = CalculateDistance(state->position, goals[i], 1.0, 1.0);
        KDL::Frame resulting_position( KDL::Rotation::RPY(velocities[3]*action_timestep_, velocities[4]*action_timestep_, 
            velocities[5]*action_timestep_) * state->position.M, 
            state->position.p + KDL::Vector(velocities[0]*action_timestep_, velocities[1]*action_timestep_, velocities[2]*action_timestep_));
        double dist_after_move = CalculateDistance(resulting_position, goals[i], 1.0, 1.0);
        rewards.push_back(exp(dist_factor*(dist_to_start - dist_after_move)));
        reward_sum += rewards[i];
    }
    
    // Calculate probability based on softmax over reward for each goal
    for(int i = 0; i < goals.size(); ++i)
    {
        double goal_prob = rewards[i] / reward_sum;
        log_likelihoods[i] += log(goal_prob);
    }
    probs_updated_since_last_calc = true;
}

void GoalClassifierUserModel::GetProbabilities(std::vector<double>& probabilities_out) const
{
    // Calculates current normalized probabilities based on likelihoods of each goal
    probabilities_out.clear();
    if(probs_updated_since_last_calc)
    {
        CalculateProbabilities();
    }
    goal_prob_lock.lock();
    for(int i = 0; i < goal_probabilities.size(); ++i)
    {
        probabilities_out.push_back(goal_probabilities[i]);
    }
    goal_prob_lock.unlock();
}

void GoalClassifierUserModel::CalculateProbabilities() const
{
    if(log_likelihoods.size() == 0) return;

    double total_log_prob = log_likelihoods[0];
    for(int i = 1; i < log_likelihoods.size(); ++i)
    {
        // Based on Numpy's logaddexp to add
        // Source: https://github.com/horta/logaddexp/blob/main/include/logaddexp/logaddexp.h
        double tmp = total_log_prob - log_likelihoods[i];
        if(total_log_prob == log_likelihoods[i])
        {
            total_log_prob = total_log_prob + log(2.0);
        } 
        else if (tmp > 0)
        {
            total_log_prob = total_log_prob + log1p(exp(-tmp));
        } 
        else 
        {
            total_log_prob = log_likelihoods[i] + log1p(exp(tmp));
        }
    }

    goal_prob_lock.lock();
    goal_probabilities.clear();
    for(int i = 0; i < log_likelihoods.size(); ++i)
    {
        goal_probabilities.push_back(exp(log_likelihoods[i] - total_log_prob));
    }
    goal_prob_lock.unlock();
    probs_updated_since_last_calc = false;
}

double GoalClassifierUserModel::GetSampleProbability(const int sample_bias) const
{
    if(probs_updated_since_last_calc)
    {
        CalculateProbabilities();
    }
    goal_prob_lock.lock();
    double result = goal_probabilities[sample_bias];
    goal_prob_lock.unlock();
    return result;
}

std::pair<int, double> GoalClassifierUserModel::RandomSample(const boost::shared_ptr<MotionState> state, std::vector<double>& sample_out, const int sample_bias) const
{
    std::vector<double> goal_probs;
    GetProbabilities(goal_probs);
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    // Choose goal with each goal having a chance of being chosen equal to its current probability
    double random_result = random_distribution(generator);
    int sampling_goal = sample_bias;
    if(sampling_goal == -1)
    {
        sampling_goal = goal_probs.size() - 1;
        for(int j = 0; j < goal_probs.size() - 1; ++j)
        {
            random_result -= goal_probs[j];
            if(random_result <= 0)
            {
                sampling_goal = j;
                break;
            }
        }
    }

    // Create Gaussian distributions for each of the 6 velocity dimensions centered around movement toward chosen goal
    state->CalculatePosition(robot_model_);
    //Create normal distributions for each velocity primitive with mean of current velocity value and standard deviation 1/10th the size of the mean
    std::vector<std::normal_distribution<double>> velocity_distributions;
    KDL::Twist target_twist = KDL::diff(state->position, goals[sampling_goal]);
    for(int i = 0; i < 6; i++)
    {
        double target_mean;
        if(i < 3)
        {
            target_mean = target_twist.vel(i);
            
        }
        else
        {
            target_mean = target_twist.rot(i - 3);
        }
        velocity_distributions.push_back(std::normal_distribution<double>(target_mean, abs(target_mean / 10.0)));
    }

    //Generate number of velocity primitives according to set parameter
    double vel_norm = 0.0;
    //Create new velocity primitive
    std::vector<double> velocity_primitive;
    for(int j = 0; j < velocity_distributions.size(); j++)
    {
        velocity_primitive.push_back(velocity_distributions[j](generator));
        vel_norm += pow(velocity_primitive[j], 2);
    }
    vel_norm = sqrt(vel_norm);

    sample_out.clear();
    double target_norm = std::max(last_velocity_command_norm_, 0.1);
    for(int i = 0; i < velocity_primitive.size(); ++i)
    {
        sample_out.push_back((velocity_primitive[i] / vel_norm) * target_norm);
    }
    return std::pair<int, double>(sampling_goal, goal_probs[sampling_goal]);
}

}