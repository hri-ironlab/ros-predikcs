/*
Defines a state class that maintains robot joint positions and implements forward kinematics.
Built on KDL library.

Author: Connor Brooks
*/

#ifndef PREDIKCS_ROBOT_MODEL_H
#define PREDIKCS_ROBOT_MODEL_H

#include <string>
#include <vector>
#include <mutex>

#include <ros/ros.h>

#include <urdf/model.h>

#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/frames.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacdotsolver.hpp>

namespace predikcs
{

class RobotModel
{
public:
    RobotModel();
    ~RobotModel()
    {}

    void Init();

    int GetNumberOfJoints() const;

    void GetJacobian(std::vector<double> joint_positions, Eigen::Matrix<double,6,Eigen::Dynamic>* jac) const;

    void GetJacobianDot(std::vector<double> joint_positions, int joint_index, Eigen::Matrix<double,6,Eigen::Dynamic>* jac_dot) const;

    void GetPosition(std::vector<double> joint_positions, KDL::Frame* position) const;

    double GetJointPosUpLimit(const int joint_index) const;

    double GetJointPosDownLimit(const int joint_index) const;

    double GetJointVelLimit(const int joint_index) const;

private:
    bool busy_;
    bool initialized_;

    std::vector<double> jnt_pos_up_limits_;
    std::vector<double> jnt_pos_down_limits_;
    std::vector<double> jnt_vel_limits_;

    std::string root_link_;
    std::string tip_link_;

    urdf::Model model_;

    KDL::Chain kdl_chain_;
    std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;
    mutable std::mutex jac_solver_lock_;
    std::shared_ptr<KDL::ChainJntToJacDotSolver> jac_dot_solver_;
    mutable std::mutex jac_dot_solver_lock_;
    std::shared_ptr<KDL::ChainFkSolverPos_recursive> jnt_to_pos_solver_;
    mutable std::mutex jnt_to_pos_solver_lock_;
};

}

#endif  // PREDIKCS_ROBOT_MODEL_H