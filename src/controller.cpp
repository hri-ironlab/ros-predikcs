#include <ros/ros.h>
#include <vector>
#include <boost/format.hpp>
#include <boost/pointer_cast.hpp>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <control_msgs/GripperCommandActionGoal.h>
#include <string>
#include <std_msgs/String.h>
#include <sensor_msgs/JointState.h>
#include <yaml-cpp/yaml.h>
#include "predikcs/states/robot_model.h"
#include "predikcs/user_models/user_model.h"
#include "predikcs/user_models/goal_classifier_user_model.h"
#include "predikcs/search/reward_calculator.h"
#include "predikcs/states/motion_state.h"
#include "predikcs/search/voo_bandit.h"

class Controller
{
public:
    Controller(ros::NodeHandle& nh, double loop_rate) : voo_spec(new predikcs::VooSpec), robot(new predikcs::RobotModel),
    user(new predikcs::UserModel(0,0)), reward(new predikcs::RewardCalculator)
    {
        joint_sub = nh.subscribe<sensor_msgs::JointState>("joint_states", 1, boost::bind(&Controller::JointUpdateCallback, this, _1));
        vel_command_sub = nh.subscribe<geometry_msgs::Twist>("teleop_commands", 1, boost::bind(&Controller::VelocityCommandCallback, this, _1));
        gripper_command_sub = nh.subscribe<control_msgs::GripperCommandActionGoal>("/gripper_controller/gripper_action/goal", 1, boost::bind(&Controller::GripperCommandCallback, this, _1));
        command_pub = nh.advertise<sensor_msgs::JointState>("joint_commands", 1);

        // Set search parameters. These can be defined as needed but should be tuned for your use case
        // Longer rollouts will typically provide smoother movement for a given time interval, provided there is enough computational resources to gather a large number of samples (100+ active samples at any time)
        // Reward parameters can be configured to emphasize different components (or you can define a reward model that is entirely custom!)
        ReadParams();
        voo_spec->sampling_loop_rate = loop_rate;
        voo_spec->sampling_time_limit = loop_rate - (loop_rate / 10.0);
        voo_spec->max_voronoi_samples = kMaxVoronoiSamples;
        voo_bandit = boost::shared_ptr<predikcs::VooBandit>(new predikcs::VooBandit(voo_spec, robot, reward));

        last_velocity_command = std::vector<double>(6, 0);
        last_joint_positions = std::vector<double>(no_of_joints, 0);
        last_joint_velocities = std::vector<double>(no_of_joints, 0);
        last_joint_accelerations = std::vector<double>(no_of_joints, 0);
        current_fetch_command_msg.velocity = std::vector<double>(no_of_joints, 0);
        last_joint_msg_time = 0;
        max_accel_factor = 4.0;
        vel_command_waiting = false;
    }

    void ReadParams()
    {
        ros::param::get("/KCS_Controller/number_of_joints", no_of_joints);
        joint_names.clear();
        std::string new_joint_name;
        double new_joint_limit;
        for(int i = 0; i < no_of_joints; ++i)
        {
            ros::param::get("/KCS_Controller/joint" + std::to_string(i) + "_name", new_joint_name);
            joint_names.push_back(new_joint_name);
        }

        ros::param::get("/KCS_Controller/voo_spec/uniform_sample_prob", voo_spec->uniform_sample_prob);
        ros::param::get("/KCS_Controller/voo_spec/tau", voo_spec->tau);
        ros::param::get("/KCS_Controller/voo_spec/sample_rollouts", voo_spec->sample_rollouts);
        ros::param::get("/KCS_Controller/voo_spec/rollout_steps", voo_spec->rollout_steps);
        ros::param::get("/KCS_Controller/voo_spec/delta_t", voo_spec->delta_t);
        ros::param::get("/KCS_Controller/voo_spec/temporal_discount", voo_spec->gamma);
        double dist_weight, jerk_weight, manip_weight, lim_weight;
        ros::param::get("/KCS_Controller/reward_params/distance", dist_weight);
        ros::param::get("/KCS_Controller/reward_params/jerk", jerk_weight);
        ros::param::get("/KCS_Controller/reward_params/manipulability", manip_weight);
        ros::param::get("/KCS_Controller/reward_params/limits", lim_weight);
        reward->SetParameters(dist_weight, jerk_weight, manip_weight, lim_weight);

        boost::shared_ptr<predikcs::GoalClassifierUserModel> goal_user(new predikcs::GoalClassifierUserModel(1, voo_spec->delta_t));
        goal_user->SetRobotModel(robot);
        int num_goals;
        ros::param::get("KCS_Controller/num_goals", num_goals);
        std::vector<std::vector<double>> goals;
        for(int i = 0; i < num_goals; ++i)
        {
            std::vector<double> goal_points;
            ros::param::get("/KCS_Controller/goal_" + std::to_string(i + 1), goal_points);
            goals.push_back(goal_points);
        }
        goal_user->SetGoals(goals);
        user = goal_user;
    }

    void JointUpdateCallback(const sensor_msgs::JointState::ConstPtr& msg)
    {
        std::vector<double> new_joint_positions;
        std::vector<double> new_joint_velocities;
        std::vector<double> new_joint_accelerations;
        double joint_msg_time = msg->header.stamp.now().toSec();
        if(joint_msg_time - last_joint_msg_time < kJointUpdateTime || msg->position.size() < joint_names.size())
        {
            return;
        }
        int j = 0;
        for(int i = 0; i < msg->position.size(); ++i)
        {
            if(j < last_joint_positions.size() && msg->name[i].compare(joint_names[j]) == 0)
            {
                new_joint_positions.push_back(msg->position[i]);
                double joint_dist = new_joint_positions[j] - last_joint_positions[j];
                if(abs(joint_dist) > M_PI && robot->GetJointPosUpLimit(j) == std::numeric_limits<double>::infinity())
                {
                    joint_dist = copysign(abs(joint_dist - (2 * M_PI)), -1 * joint_dist);
                }
                new_joint_velocities.push_back(joint_dist / (joint_msg_time - last_joint_msg_time));
                new_joint_accelerations.push_back((new_joint_velocities[j] - last_joint_velocities[j]) / (joint_msg_time - last_joint_msg_time));
                ++j;
            }
        }
        if(new_joint_positions.size() != joint_names.size())
        {
            // Bad message, discard
            ROS_ERROR("Improperly sized joint positions (missing joint names)");
            return;
        }
        last_joint_positions = new_joint_positions;
        last_joint_velocities = new_joint_velocities;
        last_joint_accelerations = new_joint_accelerations;
        last_joint_msg_time = joint_msg_time;
    }

    void VelocityCommandCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        std::vector<double> new_velocity_command;
        new_velocity_command.push_back(msg->linear.x);
        new_velocity_command.push_back(msg->linear.y);
        new_velocity_command.push_back(msg->linear.z);
        new_velocity_command.push_back(msg->angular.x);
        new_velocity_command.push_back(msg->angular.y);
        new_velocity_command.push_back(msg->angular.z);
        last_velocity_command = new_velocity_command;
        vel_command_waiting = true;
    }

    void GripperCommandCallback(const control_msgs::GripperCommandActionGoal::ConstPtr& msg)
    {
        // Reset probabilities of grasp goals whenever a grasp command is received
        static_cast<predikcs::GoalClassifierUserModel*>(user.get())->ResetProbabilities();
    }

    void DecayCommand()
    {
        // Exponentially decay velocity command
        for(int i = 0; i < current_fetch_command_msg.velocity.size(); ++i)
        {
            if(abs(current_fetch_command_msg.velocity[i]) < kMinVelCommand)
            {
                current_fetch_command_msg.velocity[i] = 0.0;
            }
            else
            {
                current_fetch_command_msg.velocity[i] = current_fetch_command_msg.velocity[i] / kDecayFactor;
            }
        }
    }

    void GetNewCommand()
    {
        bool all_zeros = true;
        for(int i = 0; i < last_velocity_command.size(); ++i)
        {
            if(last_velocity_command[i] != 0.0)
            {
                all_zeros = false;
                break;
            }
        }

        if(all_zeros || !vel_command_waiting)
        {
            DecayCommand();
        }

        user->SetLastVelocityCommand(last_velocity_command);

        // Find best current null movement
        boost::shared_ptr<predikcs::MotionState> current_state( new predikcs::MotionState(last_joint_positions, last_joint_velocities, last_joint_accelerations) );
        
        current_state->CalculatePosition(robot);
        double quat_x, quat_y, quat_z, quat_w;
        current_state->position.M.GetQuaternion(quat_x, quat_y, quat_z, quat_w);
        
        voo_bandit->GetCurrentBestJointVelocities(current_state, last_velocity_command, current_fetch_command_msg.velocity);
        static_cast<predikcs::GoalClassifierUserModel*>(user.get())->UpdateProbabilities(current_state, last_velocity_command);
          
        PublishCommand();
        vel_command_waiting = false;
    }

    void GenerateNewSamples()
    {
        // Add more samples while waiting for next command
        boost::shared_ptr<predikcs::MotionState> next_state( new predikcs::MotionState(last_joint_positions, last_joint_velocities, current_fetch_command_msg.velocity, voo_spec->sampling_loop_rate, robot, 0.0) );
        
        voo_bandit->GenerateSamples(next_state, user);
    }

    void TrimAccel()
    {
        double max_accel = max_accel_factor * voo_spec->sampling_loop_rate;
        for(int i = 0; i < current_fetch_command_msg.velocity.size(); ++i)
        {
            if(current_fetch_command_msg.velocity[i] == 0.0)
            {
                continue;
            }
            if(abs(current_fetch_command_msg.velocity[i] - last_joint_velocities[i]) > max_accel)
            {
                current_fetch_command_msg.velocity[i] = last_joint_velocities[i] + copysign(max_accel, current_fetch_command_msg.velocity[i] - last_joint_velocities[i]);
            }
        }        
    }

    void PublishCommand()
    {
        TrimAccel();
        command_pub.publish(current_fetch_command_msg);
    }

    boost::shared_ptr<predikcs::VooSpec> voo_spec;
    boost::shared_ptr<predikcs::VooBandit> voo_bandit;

private:
    static constexpr double kJointUpdateTime = 0.025;
    static constexpr double kDecayFactor = 1.25;
    static constexpr double kMinVelCommand = 0.01;
    static constexpr int kMaxVoronoiSamples = 10;

    int no_of_joints;
    ros::Subscriber joint_sub;
    ros::Subscriber vel_command_sub;
    ros::Subscriber gripper_command_sub;
    ros::Publisher command_pub;
    std::vector<double> last_joint_positions;
    std::vector<double> last_joint_velocities;
    std::vector<double> last_joint_accelerations;
    double last_joint_msg_time;
    double max_accel_factor;
    std::vector<double> last_velocity_command;
    std::vector<std::string> joint_names;
    std::vector<double> joint_vel_limits;
    boost::shared_ptr<predikcs::RobotModel> robot;
    boost::shared_ptr<predikcs::UserModel> user;
    boost::shared_ptr<predikcs::RewardCalculator> reward;
    bool vel_command_waiting;
    std::string controller_spec;
    sensor_msgs::JointState current_fetch_command_msg;
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "KCS_Controller");

    ros::NodeHandle nh;
    ros::Duration(1.0).sleep();

    // Set the control loop frequency in Hz. Note that a fairly simple change is to run the sampling algorithms
    // in one thread and have a separate thread checking the current-best nullspace motion, allowing the output
    // to instead be event-based on new command messages rather than at a fixed control rate. Here, we implement
    // with a fixed control rate for the sake of a fair comparison with other controllers during experimentation.
    double kLoopFrequency = 10;

    Controller controller(nh, 1.0 / (double) kLoopFrequency);
    ros::Duration(1.0).sleep();
    ros::AsyncSpinner spinner(1);
    ros::Rate loop_rate(kLoopFrequency);
    spinner.start();

    while(ros::ok())
    {
        ros::spinOnce();
        controller.GetNewCommand();
        controller.GenerateNewSamples();
        ros::spinOnce();
        loop_rate.sleep();
    }
    spinner.stop();
}