/*
Defines a base class for user models for use in Iron Lab's Predictive Velocity Controller.
Subclasses from this base class should implement specific types of probabilistic user models.

Author: Connor Brooks
*/

#ifndef PREDIKCS_USER_MODEL_H
#define PREDIKCS_USER_MODEL_H

//includes
#include <vector>
#include <utility>
#include <random>
#include <boost/shared_ptr.hpp>

namespace predikcs
{

// forward declares
class RobotModel;
class MotionState;

class UserModel
{
public:
    UserModel(const int num_options, const double action_timestep);
    
    virtual ~UserModel()
    {}

    virtual std::pair<int, double> RandomSample(const boost::shared_ptr<MotionState> state, std::vector<double>& sample, const int sample_bias) const;

    virtual double GetSampleProbability(const int sample_bias) const { return 1.0; }

    void SetNumOptions(const int num_options){ num_options_ = num_options; }

    void SetActionTimestep(const double action_timestep){ action_timestep_ = action_timestep; }

    void SetLastVelocityCommand(const std::vector<double>& last_velocity_command) { 
        last_velocity_command_ = last_velocity_command;
    }

protected:
    int num_options_;
    double action_timestep_;
    std::vector<double> last_velocity_command_;
    double last_velocity_command_norm_;
    std::random_device rd;
    std::default_random_engine random_generator;
    std::uniform_real_distribution<double> random_distribution;
};

}

#endif  // PREDIKCS_USER_MODEL_H