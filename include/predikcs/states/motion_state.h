/*
Defines a class for packaging the details of a robot state including position, velocity, and calculated values such as the Jacobian.

Author: Connor Brooks
*/

#ifndef PREDIKCS_MOTION_STATE_H
#define PREDIKCS_MOTION_STATE_H

//includes
#include <vector>
#include <Eigen/Core>
#include <kdl/frames.hpp>
#include <predikcs/states/robot_model.h>
#include <cmath>

namespace predikcs
{

class MotionState
{
public:
    // Constructor takes in a current joint position and the last velocity applied
    MotionState(std::vector<double> last_joint_positions, std::vector<double> last_joint_velocities, std::vector<double> last_joint_accelerations)
    {
        joint_positions = last_joint_positions;
        joint_velocities = last_joint_velocities;
        joint_accelerations = last_joint_accelerations;
        is_jacobian_calculated = false;
        is_position_calculated = false;
        is_manipulability_calculated = false;
        time_in_future = 0.0;
    }

    // Constructor takes in a pointer to the starting joint positions, a set of old and new joint velocities and a timestamp over which to spin up toward and apply the new joint velocities
    MotionState(std::vector<double>* starting_joint_positions, std::vector<double>* last_joint_velocities, std::vector<double>* given_joint_velocities, double motion_time, boost::shared_ptr<RobotModel> robot_model, double time_into_future)
    {
        double timestamp, time_increment, max_acceleration, current_velocity, current_position, target_velocity;
        time_increment = 0.01;
        max_acceleration = 0.04; //Panda: 0.25
        time_in_future = time_into_future;
        for(int j = 0; j < (*starting_joint_positions).size(); j++)
        {
            timestamp = 0.0;
            double current_velocity = (*last_joint_velocities)[j];
            double current_position = (*starting_joint_positions)[j];
            double target_velocity = (*given_joint_velocities)[j];
            if(abs(target_velocity) > robot_model->GetJointVelLimit(j))
            {
                target_velocity = copysign(robot_model->GetJointVelLimit(j), target_velocity);
            }
            while(timestamp < motion_time)
            {
                current_position += current_velocity * time_increment;
                if(abs(target_velocity - current_velocity) > max_acceleration)
                {
                    current_velocity += copysign(max_acceleration, target_velocity - current_velocity);
                } else {
                    current_velocity = target_velocity;
                }
                timestamp += time_increment;
            }
            
            joint_velocities.push_back(current_velocity);
            joint_accelerations.push_back((joint_velocities[j] - (*last_joint_velocities)[j]) / motion_time);
            joint_positions.push_back(current_position);
            commanded_velocities.push_back(target_velocity);

            if(abs(joint_positions[j]) > M_PI && robot_model->GetJointPosUpLimit(j) == std::numeric_limits<double>::infinity())
            {
                if(joint_positions[j] > 0)
                {
                    joint_positions[j] = -M_PI + (joint_positions[j] - M_PI);
                }
                else
                {
                    joint_positions[j] = M_PI - (-M_PI - joint_positions[j]);
                }
            }
            else if(joint_positions[j] > robot_model->GetJointPosUpLimit(j))
            {
                joint_positions[j] = robot_model->GetJointPosUpLimit(j);
            }
            else if(joint_positions[j] < robot_model->GetJointPosDownLimit(j))
            {
                joint_positions[j] = robot_model->GetJointPosDownLimit(j);
            }
        }
        is_jacobian_calculated = false;
        is_position_calculated = false;
        is_manipulability_calculated = false;
    }
    ~MotionState()
    {}

    void CalculateJacobian(boost::shared_ptr<RobotModel> robot_model) const
    {
        if(is_jacobian_calculated)
        {
            return;
        }
        robot_model->GetJacobian(joint_positions, &jacobian);
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> jacobian_transpose = jacobian.transpose();
        pseudo_inverse = jacobian_transpose * (jacobian * jacobian_transpose).inverse();
        is_jacobian_calculated = true;
    }

    void CalculatePosition(boost::shared_ptr<RobotModel> robot_model) const
    {
        if(is_position_calculated)
        {
            return;
        }
        robot_model->GetPosition(joint_positions, &position);
        is_position_calculated = true;
    }

    void CalculateManipulability(boost::shared_ptr<RobotModel> robot_model) const
    {
        if(is_manipulability_calculated)
        {
            return;
        }
        if(!is_jacobian_calculated)
        {
            CalculateJacobian(robot_model);
        }
        manipulability = sqrt((jacobian * jacobian.transpose()).determinant());
    }

    // Resulting positions after applying joint velocities over given time period
    std::vector<double> joint_positions;
    std::vector<double> joint_velocities;
    std::vector<double> joint_accelerations;
    std::vector<double> commanded_velocities;
    double time_in_future = 0.0;

    // Values cached on calculations
    mutable bool is_jacobian_calculated;
    mutable Eigen::Matrix<double,6,Eigen::Dynamic> jacobian;
    mutable Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> pseudo_inverse;
    mutable bool is_manipulability_calculated;
    mutable double manipulability = 0.0;
    mutable bool is_position_calculated;
    mutable KDL::Frame position;
};

}

#endif  // PREDIKCS_MOTION_STATE_H