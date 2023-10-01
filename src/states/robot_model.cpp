/*
Defines a model class that handles a URDF of a robot used in planning.
Robot joint positions can be passed in for forward and inverse kinematics.
Built on KDL library.

Author: Connor Brooks
*/
#include "predikcs/states/robot_model.h"
#include <ros/ros.h>
#include <string>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

namespace predikcs
{

RobotModel::RobotModel()
{
    busy_ = true;
    initialized_ = false;
    Init();
}

void RobotModel::Init()
{
    //Get root and tip joints for planning
    ros::param::get("/KCS_Controller/planning_root_link", root_link_);
    ros::param::get("/KCS_Controller/planning_tip_link", tip_link_);

    //Load URDF from parameter server
    std::string urdf_string;
    ros::param::get("/planning_robot_urdf", urdf_string);
    model_.initString(urdf_string);

    //Load KDL tree
    KDL::Tree kdl_tree;
    kdl_parser::treeFromUrdfModel(model_, kdl_tree);
    //Populate the chain
    kdl_tree.getChain(root_link_, tip_link_, kdl_chain_);

    jac_solver_ = std::make_shared<KDL::ChainJntToJacSolver>(kdl_chain_);
    jac_dot_solver_ = std::make_shared<KDL::ChainJntToJacDotSolver>(kdl_chain_);
    jnt_to_pos_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(kdl_chain_);

    double new_joint_limit;
    for(int i = 0; i < kdl_chain_.getNrOfJoints(); ++i)
    {
        ros::param::get("/KCS_Controller/joint" + std::to_string(i) + "_pos_up_limit", new_joint_limit);
        jnt_pos_up_limits_.push_back(new_joint_limit);
        
        ros::param::get("/KCS_Controller/joint" + std::to_string(i) + "_pos_down_limit", new_joint_limit);
        jnt_pos_down_limits_.push_back(new_joint_limit);

        ros::param::get("/KCS_Controller/joint" + std::to_string(i) + "_vel_limit", new_joint_limit);
        jnt_vel_limits_.push_back(new_joint_limit);
        ROS_INFO("joint %d velocity limit: %.2f", i, jnt_vel_limits_[i]);
    }

    initialized_ = true;
    busy_ = false;
}

int RobotModel::GetNumberOfJoints() const
{
    if(!initialized_){
        return -1;
    }

    return kdl_chain_.getNrOfJoints();
}

void RobotModel::GetJacobian(const std::vector<double>& joint_positions, Eigen::Matrix<double,6,Eigen::Dynamic>& jac_out) const
{
    KDL::JntArray jnt_pos;
    KDL::Jacobian jacobian;
    jnt_pos.resize(kdl_chain_.getNrOfJoints());
    jacobian.resize(kdl_chain_.getNrOfJoints());
    //Update joint positions
    for(size_t i = 0; i < joint_positions.size(); i++)
    {
        jnt_pos(i) = joint_positions[i];
    }

    jac_solver_lock_.lock();
    jac_solver_->JntToJac(jnt_pos, jacobian);
    jac_solver_lock_.unlock();

    jac_out.resize(jacobian.rows(), jacobian.columns());
    for(int i = 0; i < jacobian.rows(); ++i)
    {
        for(int j = 0; j < jacobian.columns(); ++j)
        {
            jac_out(i, j) = jacobian(i, j);
        }
    }
}

void RobotModel::GetJacobianDot(const std::vector<double>& joint_positions, const int joint_index, Eigen::Matrix<double,6,Eigen::Dynamic>& jac_dot_out) const
{
    KDL::Jacobian jacobian_dot;
    jacobian_dot.resize(kdl_chain_.getNrOfJoints());
    KDL::JntArray joint_pos(joint_positions.size());
    KDL::JntArray joint_vel(joint_positions.size());
    for(int i = 0; i < joint_positions.size(); ++i) {
        joint_pos(i) = joint_positions[i];
        if(i == joint_index) {
            joint_vel(i) = 1.0;
        } else {
            joint_vel(i) = 0.0;
        }
    }
    KDL::JntArrayVel system_state(joint_pos, joint_vel);
    jac_dot_solver_lock_.lock();
    jac_dot_solver_->JntToJacDot(system_state, jacobian_dot);
    jac_dot_solver_lock_.unlock();

    jac_dot_out.resize(jacobian_dot.rows(), jacobian_dot.columns());
    for(int i = 0; i < jacobian_dot.rows(); ++i)
    {
        for(int j = 0; j < jacobian_dot.columns(); ++j)
        {
            jac_dot_out(i, j) = jacobian_dot(i, j);
        }
    }
}

void RobotModel::GetPosition(const std::vector<double>& joint_positions, KDL::Frame& position_out) const
{
    KDL::JntArray jnt_pos;
    jnt_pos.resize(joint_positions.size());
    //Update joint positions
    for(size_t i = 0; i < joint_positions.size(); i++)
    {
        jnt_pos(i) = joint_positions[i];
    }
    jnt_to_pos_solver_lock_.lock();
    jnt_to_pos_solver_->JntToCart(jnt_pos, position_out);
    jnt_to_pos_solver_lock_.unlock();
}

double RobotModel::GetJointPosUpLimit(const int joint_index) const
{
    return jnt_pos_up_limits_[joint_index];
}

double RobotModel::GetJointPosDownLimit(const int joint_index) const
{
    return jnt_pos_down_limits_[joint_index];
}

double RobotModel::GetJointVelLimit(const int joint_index) const
{
    return jnt_vel_limits_[joint_index];
}

}