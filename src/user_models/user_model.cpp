//Base version of user model. This class implements a user model that models the user as exponentially more likely to choose velocities the closer they are to the last velocity choice.
//Subclasses of this user model class should implement other strategies for predicting user movement selection.

//includes
#include "predikcs/user_models/user_model.h"
#include <cmath>
#include <ros/console.h>
#include <chrono>

namespace predikcs
{

//----------------------------------------------------------------------------------------------------------------------------
// Utility functions for this version of user model

//Calculates velocity difference as the L2 norm of the acceleration going from one velocity to the other
double CalculateVelocityDiff(std::vector<double>* old_velocities, std::vector<double>* new_velocities)
{
    double sum_squared_diffs = 0.0;
    for(int i = 0; i < new_velocities->size(); i++)
    {
        sum_squared_diffs += pow(((*new_velocities)[i] - (*old_velocities)[i]),2);
    }

    return std::max(sqrt(sum_squared_diffs), 0.1);
}

void GenerateActionOptions(std::vector<std::vector<double>>* velocity_primitives, std::vector<double> last_velocity_command_, int num_options_)
{
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    //Create normal distributions for each velocity primitive with mean of current velocity value and standard deviation of 0.1
    std::vector<std::normal_distribution<double>> velocity_distributions;
    for(int i = 0; i < last_velocity_command_.size(); i++)
    {
        velocity_distributions.push_back(std::normal_distribution<double>(last_velocity_command_[i], 0.1));
    }

    //Generate number of velocity primitives according to set parameter
    for(int i = 0; i < num_options_; i++)
    {
        //Create new velocity primitive
        std::vector<double> velocity_primitive;
        for(int j = 0; j < velocity_distributions.size(); j++)
        {
            velocity_primitive.push_back(velocity_distributions[j](generator));
        }
        velocity_primitives->push_back(velocity_primitive);
    }
}

//----------------------------------------------------------------------------------------------------------------------------
// UserModel definition

UserModel::UserModel(const int num_options, const double action_timestep) : random_generator(rd()), random_distribution(0.0, 1.0)
{ 
    SetNumOptions(num_options);
    SetActionTimestep(action_timestep);
    for(int i = 0; i < 6; ++i)
    {
        last_velocity_command_.push_back(0.0);
    }
}

std::pair<int, double> UserModel::RandomSample(const boost::shared_ptr<MotionState> state, std::vector<double>& sample, const int sample_bias) const
{
    std::vector<std::vector<double>> movement_options;
    GenerateActionOptions(&movement_options, last_velocity_command_, 1);
    
    sample.clear();
    for(int i = 0; i < movement_options[0].size(); ++i)
    {
        sample.push_back(movement_options[0][i]);
    }
    return std::pair<int, double>(-1, 1.0);
}

}