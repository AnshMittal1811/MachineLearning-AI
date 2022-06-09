#pragma once

#include <stdlib.h>
#include <set>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>

namespace raisim {

struct Node {
  int id;
  std::vector<Node*> neighbour;
  std::vector<int> neighbour_weight;
  std::vector<int> original_weight;
  int max_weight;
  int weight;
};

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    walker_ = world_->addArticulatedSystem(resourceDir_+"/humanoid.urdf");
    walker_->setName("walker");
    walker_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround(0, "steel");
    world_->setERP(1.0);

    world_->setMaterialPairProp("default", "ball", 1.0, 0.8, 0.0001);
    world_->setMaterialPairProp("default", "steel", 5.0, 0.0, 0.0001);
    soccer_ = world_->addArticulatedSystem(resourceDir_+"/ball3D.urdf");
    soccer_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
    ball_gc_init_.setZero(7);
    ball_gv_init_.setZero(6);
    ball_gc_init_[3] = 1;
    ball_gc_.setZero(7);
    ball_gc_[3] = 1;
    ball_gv_.setZero(6);
    ball_gc_init_[0] = 0.2;
    ball_gc_init_[1] = 0.15;
    ball_reference_.setZero(7);
    ball_reference_vel_.setZero(6);
    soccer_->setState(ball_gc_init_, ball_gv_init_);

    /// get robot data
    gcDim_ = walker_->getGeneralizedCoordinateDim();
    gvDim_ = walker_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    reference_.setZero(gcDim_);

    around_the_world_reference_ = Eigen::MatrixXd::Zero(30, 43);
    around_the_world_ball_reference_ = Eigen::MatrixXd::Zero(30, 3);
    read_around_the_world();

    around_the_world_right_reference_ = Eigen::MatrixXd::Zero(30, 43);
    around_the_world_ball_right_reference_ = Eigen::MatrixXd::Zero(30, 3);
    read_around_the_world_right();

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;
    reference_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(250.0);//250
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(25.);//25
    jointPgain.segment(9, 3).setConstant(50.0); jointDgain.segment(9, 3).setConstant(5.0);
    jointPgain.segment(12, 8).setConstant(100.0); jointDgain.segment(12, 8).setConstant(10.0);
    jointPgain.segment(24, 3).setConstant(150.0); jointDgain.segment(24, 3).setConstant(15.0);
    jointPgain.segment(31, 3).setConstant(150.0); jointDgain.segment(31, 3).setConstant(15.0);
    jointPgain.segment(12, 8).setConstant(50.0); jointDgain.segment(12, 8).setConstant(5.0);
    walker_->setPdGains(jointPgain, jointDgain);
    walker_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));


    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 88 + 3 * num_task_;//94;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    stateDim_ = gcDim_ + 7;
    stateDouble_.setZero(stateDim_ * 2);

    /// action scaling
    //actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(1);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    task_vector_.setZero(num_task_), next_task_vector_.setZero(num_task_), next_next_task_vector_.setZero(num_task_);

    //initialize control graph
    psuduo_node_->id = -1;
    psuduo_node_->max_weight = 0;

    int id = 0;
    foot_juggle_down_node_->id = id; //0
    foot_juggle_down_node_->weight = 1;
    id++;
    foot_juggle_up_node_->id = id; // 1
    foot_juggle_up_node_->weight = 1;
    id++;
    around_the_world_up_node_->id = id; //2
    around_the_world_up_node_->weight = 1;
    id++;
    around_the_world_down_node_->id = id; //3
    around_the_world_down_node_->weight = 1;
    id++;
    foot_stall_enter_node_->id = id; //4
    foot_stall_enter_node_->weight = 1;
    id++;
    foot_stall_exit_node_->id = id;// 5
    foot_stall_exit_node_->weight = 1;
    id++;
    chest_stall_node_->id = id; //6
    chest_stall_node_->weight = 1;
    id++;
    chest_juggle_up_node_->id = id; //7
    chest_juggle_up_node_->weight = 1;
    id++;
    chest_juggle_down_node_->id = id; //8
    chest_juggle_down_node_->weight = 1;
    id++;
    head_juggle_down_node_->id = id; //9
    head_juggle_down_node_->weight = 1;
    id++;
    head_juggle_up_node_->id = id; //10
    head_juggle_up_node_->weight = 1;
    id++;
    head_stall_node_->id = id; //11
    head_stall_node_->weight = 1;
    id++;
    head_stall_exit_node_->id = id; //12
    head_stall_exit_node_->weight = 1;
    id++;
    knee_juggle_down_node_->id = id;//13
    knee_juggle_down_node_->weight = 1;
    id++;
    knee_juggle_up_node_->id = id; //14
    knee_juggle_up_node_->weight = 1;
    id++;
    right_foot_juggle_down_node_->id = id; //15
    right_foot_juggle_down_node_->weight = 1;
    id++;
    right_foot_juggle_up_node_->id = id; //16
    right_foot_juggle_up_node_->weight = 1;
    id++;
    right_knee_juggle_down_node_->id = id; //17
    right_knee_juggle_down_node_->weight = 1;
    id++;
    right_knee_juggle_up_node_->id = id; //18
    right_knee_juggle_up_node_->weight = 1;
    id++;
    right_around_the_world_down_node_->id = id; //19
    right_around_the_world_down_node_->weight = 1;
    id++;
    right_around_the_world_up_node_->id = id; //20
    right_around_the_world_up_node_->weight = 1;
    id++;
    right_foot_stall_node_->id = id; // 21
    right_foot_stall_node_->weight = 1;
    id++;
    right_foot_stall_exit_node_->id = id; //22
    right_foot_stall_exit_node_->weight = 1;

    foot_juggle_down_node_->neighbour.push_back(foot_juggle_up_node_);
    foot_juggle_down_node_->neighbour_weight.push_back(1);
    foot_juggle_down_node_->original_weight.push_back(1);
    foot_juggle_down_node_->neighbour.push_back(around_the_world_up_node_);
    foot_juggle_down_node_->neighbour_weight.push_back(1);
    foot_juggle_down_node_->original_weight.push_back(1);
    // foot_juggle_down_node_->neighbour.push_back(foot_stall_enter_node_);
    // foot_juggle_down_node_->neighbour_weight.push_back(1);
    // foot_juggle_down_node_->original_weight.push_back(1);
    foot_juggle_down_node_->max_weight = 2;
    
    foot_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    foot_juggle_up_node_->neighbour_weight.push_back(1);
    foot_juggle_up_node_->original_weight.push_back(1);
    // foot_juggle_up_node_->neighbour.push_back(chest_juggle_down_node_);
    // foot_juggle_up_node_->neighbour_weight.push_back(1);
    // foot_juggle_up_node_->original_weight.push_back(1);
    foot_juggle_up_node_->neighbour.push_back(head_juggle_down_node_);
    foot_juggle_up_node_->neighbour_weight.push_back(1);
    foot_juggle_up_node_->original_weight.push_back(1);
    foot_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    foot_juggle_up_node_->neighbour_weight.push_back(1);
    foot_juggle_up_node_->original_weight.push_back(1);
    foot_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    foot_juggle_up_node_->neighbour_weight.push_back(1);
    foot_juggle_up_node_->original_weight.push_back(1);
    foot_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    foot_juggle_up_node_->neighbour_weight.push_back(1);
    foot_juggle_up_node_->original_weight.push_back(1);
    foot_juggle_up_node_->max_weight = 5;
    
    around_the_world_up_node_->neighbour.push_back(around_the_world_down_node_);
    around_the_world_up_node_->neighbour_weight.push_back(1);
    around_the_world_up_node_->original_weight.push_back(1);
    around_the_world_up_node_->max_weight = 1;
    
    around_the_world_down_node_->neighbour.push_back(around_the_world_up_node_);
    around_the_world_down_node_->neighbour_weight.push_back(1);
    around_the_world_down_node_->original_weight.push_back(1);
    around_the_world_down_node_->neighbour.push_back(foot_juggle_up_node_);
    around_the_world_down_node_->neighbour_weight.push_back(1);
    around_the_world_down_node_->original_weight.push_back(1);
    around_the_world_down_node_->max_weight = 2;

    foot_stall_enter_node_->neighbour.push_back(foot_stall_enter_node_);
    foot_stall_enter_node_->neighbour_weight.push_back(30);
    foot_stall_enter_node_->original_weight.push_back(30);
    foot_stall_enter_node_->neighbour.push_back(foot_stall_exit_node_);
    foot_stall_enter_node_->neighbour_weight.push_back(1);
    foot_stall_enter_node_->original_weight.push_back(1);
    foot_stall_enter_node_->max_weight = 31;

    foot_stall_exit_node_->neighbour.push_back(foot_juggle_up_node_);
    foot_stall_exit_node_->neighbour_weight.push_back(1);
    foot_stall_exit_node_->original_weight.push_back(1);
    foot_stall_exit_node_->max_weight = 1;

    chest_stall_node_->neighbour.push_back(chest_stall_node_);
    chest_stall_node_->neighbour_weight.push_back(1);
    chest_stall_node_->original_weight.push_back(1);
    chest_stall_node_->max_weight = 1;

    chest_juggle_down_node_->neighbour.push_back(chest_juggle_up_node_);
    chest_juggle_down_node_->neighbour_weight.push_back(1);
    chest_juggle_down_node_->original_weight.push_back(1);
    chest_juggle_down_node_->max_weight = 1;

    chest_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    chest_juggle_up_node_->neighbour_weight.push_back(1);
    chest_juggle_up_node_->original_weight.push_back(1);
    // chest_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    // chest_juggle_up_node_->neighbour_weight.push_back(1);
    // chest_juggle_up_node_->original_weight.push_back(1);
    chest_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    chest_juggle_up_node_->neighbour_weight.push_back(1);
    chest_juggle_up_node_->original_weight.push_back(1);
    // chest_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    // chest_juggle_up_node_->neighbour_weight.push_back(1);
    // chest_juggle_up_node_->original_weight.push_back(1);
    chest_juggle_up_node_->max_weight = 2;

    head_juggle_down_node_->neighbour.push_back(head_juggle_up_node_);
    head_juggle_down_node_->neighbour_weight.push_back(1);
    head_juggle_down_node_->original_weight.push_back(1);
    // head_juggle_down_node_->neighbour.push_back(head_stall_node_);
    // head_juggle_down_node_->neighbour_weight.push_back(1);
    // head_juggle_down_node_->original_weight.push_back(1);
    head_juggle_down_node_->max_weight = 1;

    head_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    head_juggle_up_node_->neighbour_weight.push_back(1);
    head_juggle_up_node_->original_weight.push_back(1);
    head_juggle_up_node_->neighbour.push_back(head_juggle_down_node_);
    head_juggle_up_node_->neighbour_weight.push_back(1);
    head_juggle_up_node_->original_weight.push_back(1);
    head_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    head_juggle_up_node_->neighbour_weight.push_back(1);
    head_juggle_up_node_->original_weight.push_back(1);
    head_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    head_juggle_up_node_->neighbour_weight.push_back(1);
    head_juggle_up_node_->original_weight.push_back(1);
    head_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    head_juggle_up_node_->neighbour_weight.push_back(1);
    head_juggle_up_node_->original_weight.push_back(1);
    head_juggle_up_node_->max_weight = 5;

    head_stall_node_->neighbour.push_back(head_stall_node_);
    head_stall_node_->neighbour_weight.push_back(30);
    head_stall_node_->original_weight.push_back(30);
    head_stall_node_->neighbour.push_back(head_stall_exit_node_);
    head_stall_node_->neighbour_weight.push_back(1);
    head_stall_node_->original_weight.push_back(1);
    head_stall_node_->max_weight = 31;

    head_stall_exit_node_->neighbour.push_back(head_juggle_up_node_);
    head_stall_exit_node_->neighbour_weight.push_back(1);
    head_stall_exit_node_->original_weight.push_back(1);
    head_stall_exit_node_->max_weight = 1;

    knee_juggle_down_node_->neighbour.push_back(knee_juggle_up_node_);
    knee_juggle_down_node_->neighbour_weight.push_back(1);
    knee_juggle_down_node_->original_weight.push_back(1);
    knee_juggle_down_node_->max_weight = 1;

    knee_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    knee_juggle_up_node_->neighbour_weight.push_back(1);
    knee_juggle_up_node_->original_weight.push_back(1);
    knee_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    knee_juggle_up_node_->neighbour_weight.push_back(1);
    knee_juggle_up_node_->original_weight.push_back(1);
    // knee_juggle_up_node_->neighbour.push_back(chest_juggle_down_node_);
    // knee_juggle_up_node_->neighbour_weight.push_back(1);
    // knee_juggle_up_node_->original_weight.push_back(1);
    knee_juggle_up_node_->neighbour.push_back(head_juggle_down_node_);
    knee_juggle_up_node_->neighbour_weight.push_back(1);
    knee_juggle_up_node_->original_weight.push_back(1);
    knee_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    knee_juggle_up_node_->neighbour_weight.push_back(1);
    knee_juggle_up_node_->original_weight.push_back(1);
    knee_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    knee_juggle_up_node_->neighbour_weight.push_back(1);
    knee_juggle_up_node_->original_weight.push_back(1);
    knee_juggle_up_node_->max_weight = 5;

    right_foot_juggle_down_node_->neighbour.push_back(right_foot_juggle_up_node_);
    right_foot_juggle_down_node_->neighbour_weight.push_back(1);
    right_foot_juggle_down_node_->original_weight.push_back(1);
    // right_foot_juggle_down_node_->neighbour.push_back(right_foot_stall_node_);
    // right_foot_juggle_down_node_->neighbour_weight.push_back(1);
    // right_foot_juggle_down_node_->original_weight.push_back(1);
    right_foot_juggle_down_node_->neighbour.push_back(right_around_the_world_up_node_);
    right_foot_juggle_down_node_->neighbour_weight.push_back(1);
    right_foot_juggle_down_node_->original_weight.push_back(1);
    right_foot_juggle_down_node_->max_weight = 2;
    
    right_foot_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    right_foot_juggle_up_node_->original_weight.push_back(1);
    right_foot_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    right_foot_juggle_up_node_->original_weight.push_back(1);
    right_foot_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    right_foot_juggle_up_node_->original_weight.push_back(1);
    right_foot_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    right_foot_juggle_up_node_->original_weight.push_back(1);
    right_foot_juggle_up_node_->neighbour.push_back(head_juggle_down_node_);
    right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    right_foot_juggle_up_node_->original_weight.push_back(1);
    // right_foot_juggle_up_node_->neighbour.push_back(chest_juggle_down_node_);
    // right_foot_juggle_up_node_->neighbour_weight.push_back(1);
    // right_foot_juggle_up_node_->original_weight.push_back(1);
    right_foot_juggle_up_node_->max_weight = 5;

    right_knee_juggle_down_node_->neighbour.push_back(right_knee_juggle_up_node_);
    right_knee_juggle_down_node_->neighbour_weight.push_back(1);
    right_knee_juggle_down_node_->original_weight.push_back(1);
    right_knee_juggle_down_node_->max_weight = 1;

    right_knee_juggle_up_node_->neighbour.push_back(right_knee_juggle_down_node_);
    right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    right_knee_juggle_up_node_->original_weight.push_back(1);
    right_knee_juggle_up_node_->neighbour.push_back(right_foot_juggle_down_node_);
    right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    right_knee_juggle_up_node_->original_weight.push_back(1);
    right_knee_juggle_up_node_->neighbour.push_back(foot_juggle_down_node_);
    right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    right_knee_juggle_up_node_->original_weight.push_back(1);
    right_knee_juggle_up_node_->neighbour.push_back(knee_juggle_down_node_);
    right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    right_knee_juggle_up_node_->original_weight.push_back(1);
    right_knee_juggle_up_node_->neighbour.push_back(head_juggle_down_node_);
    right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    right_knee_juggle_up_node_->original_weight.push_back(1);
    // right_knee_juggle_up_node_->neighbour.push_back(chest_juggle_down_node_);
    // right_knee_juggle_up_node_->neighbour_weight.push_back(1);
    // right_knee_juggle_up_node_->original_weight.push_back(1);
    right_knee_juggle_up_node_->max_weight = 5;

    right_around_the_world_up_node_->neighbour.push_back(right_around_the_world_down_node_);
    right_around_the_world_up_node_->neighbour_weight.push_back(1);
    right_around_the_world_up_node_->original_weight.push_back(1);
    right_around_the_world_up_node_->max_weight = 1;
    
    right_around_the_world_down_node_->neighbour.push_back(right_around_the_world_up_node_);
    right_around_the_world_down_node_->neighbour_weight.push_back(1);
    right_around_the_world_down_node_->original_weight.push_back(1);
    right_around_the_world_down_node_->neighbour.push_back(right_foot_juggle_up_node_);
    right_around_the_world_down_node_->neighbour_weight.push_back(1);
    right_around_the_world_down_node_->original_weight.push_back(1);
    right_around_the_world_down_node_->max_weight = 2;

    right_foot_stall_node_->neighbour.push_back(right_foot_stall_node_);
    right_foot_stall_node_->neighbour_weight.push_back(30);
    right_foot_stall_node_->original_weight.push_back(30);
    right_foot_stall_node_->neighbour.push_back(right_foot_stall_exit_node_);
    right_foot_stall_node_->neighbour_weight.push_back(1);
    right_foot_stall_node_->original_weight.push_back(1);
    right_foot_stall_node_->max_weight = 31;

    right_foot_stall_exit_node_->neighbour.push_back(right_foot_juggle_up_node_);
    right_foot_stall_exit_node_->neighbour_weight.push_back(1);
    right_foot_stall_exit_node_->original_weight.push_back(1);
    right_foot_stall_exit_node_->max_weight = 1;

    initial_node.push_back(foot_juggle_down_node_);
    initial_node.push_back(head_juggle_down_node_);
    // initial_node.push_back(chest_juggle_down_node_);
    //initial_node.push_back(chest_stall_node_);
    initial_node.push_back(around_the_world_down_node_);
    initial_node.push_back(knee_juggle_down_node_);
    initial_node.push_back(right_foot_juggle_down_node_);
    initial_node.push_back(right_knee_juggle_down_node_);
    initial_node.push_back(right_around_the_world_down_node_);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(walker_);
    }

  }

  void init() final { }

  int select_next_node(Node* node) {
    int selection = rand() % node->max_weight;
    int index = 0;
    int current_weight = 0;
    while (selection >= node->neighbour_weight[index] + current_weight && current_weight < node->max_weight) {
      current_weight += node->neighbour_weight[index];
      index++;
    }
    if (previous_choice_ == -1) {
      previous_choice_ = index;
      previous_previous_choice_ = -1;
    }
    else if (previous_previous_choice_ == -1) {
      previous_previous_choice_ = previous_choice_;
      previous_choice_ = index;
    }
    else {
      previous_previous_previous_choice_ = previous_previous_choice_;
      previous_previous_choice_ = previous_choice_;
      previous_choice_ = index;
    }
    return index;
  }

  void update_previous_node() {
    //return;
    previous_node_->max_weight -= previous_node_->original_weight[previous_previous_previous_choice_];
    previous_node_->neighbour_weight[previous_previous_previous_choice_] -= previous_node_->original_weight[previous_previous_previous_choice_];
    if (previous_node_->neighbour_weight[previous_previous_previous_choice_] > previous_node_->original_weight[previous_previous_previous_choice_]) {
      previous_node_->max_weight -= 1;
      previous_node_->neighbour_weight[previous_previous_previous_choice_] -= 1;
    }
  }

  void update_current_node() {
    //return;
    if (current_node_->neighbour_weight[previous_previous_choice_] > 100 * current_node_->original_weight[previous_previous_choice_]) {
      int difference = (current_node_->neighbour_weight[previous_previous_choice_] - 100 * current_node_->original_weight[previous_previous_choice_]);
      current_node_->neighbour_weight[previous_previous_choice_] -= difference;
      current_node_->max_weight -= difference;
    }
    current_node_->neighbour_weight[previous_previous_choice_] += current_node_->original_weight[previous_previous_choice_];
    current_node_->max_weight += current_node_->original_weight[previous_previous_choice_];
  }

  void reset() final {
    previous_choice_ = -1, previous_previous_choice_ = -1;
    speed = 0.0;//(double(rand() % 8) - 2.0) / 10.0;
    phase_ = 0;//rand() % max_phase_;
    phase_speed_ = (4.0 + (double)rand() / RAND_MAX * 2); 
    next_phase_speed_ = (4.0 + (double)rand() / RAND_MAX * 2);
    sim_step_ = 0;
    total_reward_ = 0;

    //initialize task
    previous_node_ = psuduo_node_;
    current_node_ = initial_node[(rand() % initial_node.size())];
    next_node_ = current_node_->neighbour[select_next_node(current_node_)];
    next_next_node_ = next_node_->neighbour[select_next_node(next_node_)];

    task_vector_.setZero();
    contact_terminal_flag_ = false;
    task_vector_[current_node_->id] = 1;
    next_task_vector_.setZero();
    next_task_vector_[next_node_->id] = 1;
    next_next_task_vector_.setZero();
    next_next_task_vector_[next_next_node_->id] = 1;


    switch (current_node_->id) {
      case 5:
        desired_max_height_ = 2.0;
        break;
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
        desired_max_height_ = 2.0;
        break;
      default:
        desired_max_height_ = 1.4;   
    }
    calculate_phase_speed(desired_max_height_);
    switch (next_next_node_->id) {
      case 0:
      case 1:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
         if (current_node_->id == 7 or current_node_->id == 8 or current_node_->id == 9 or current_node_->id == 10)
          desired_max_height_ = 2.0;
         else
          desired_max_height_ = 1.4;
         break;
      case 5:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 22:
        desired_max_height_ = 2.0;
        break;
      default:
        desired_max_height_ = 1.4;
    }
    
    if (current_node_->id == 9 or current_node_->id == 10) {
      setReferenceMotionHead();
      setBallReference();
    }
    else if (current_node_->id == 0 or current_node_->id == 1 or current_node_->id == 13 
      or current_node_->id == 14 or current_node_->id == 15 or current_node_->id == 16
      or current_node_->id == 17 or current_node_->id == 18) {
      setReferenceMotion();
      setBallReference();
    }
    else if (current_node_->id == 2 or current_node_->id == 3)
      setReferenceMotionAroundTheWorld();
    else if (current_node_->id == 19 or current_node_->id == 20)
      setReferenceMotionAroundTheWorldRight();
    else if (current_node_->id == 7 or current_node_->id == 8) {
      setChestMotion();
      setBallReference();
    }

    walker_->setState(reference_, gv_init_);
    soccer_->setState(ball_reference_, ball_reference_vel_ * 0);
    updateObservation();
  }

  void setEnvironmentTask(int i) final {
    use_head_ = i;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    if (current_node_->id == 9 or current_node_->id == 10) {
      setReferenceMotionHead();
      setBallReference();
    }
    else if (current_node_->id == 0 or current_node_->id == 1 or current_node_->id == 13 or current_node_->id == 14
      or current_node_->id == 15 or current_node_->id == 16 or current_node_->id == 17 or current_node_->id == 18){
      setReferenceMotion();
      setBallReference();
    }
    else if (current_node_->id == 2 or current_node_->id == 3) {
      setReferenceMotionAroundTheWorld();
      phase_speed_ = max_phase_ / 30;
    }
    else if (current_node_->id == 19 or current_node_->id == 20) {
      setReferenceMotionAroundTheWorldRight();
      phase_speed_ = max_phase_ / 30;
    }
    else if (current_node_->id == 4 || current_node_->id == 5 || current_node_->id == 21 || current_node_->id == 22) {
      setReferenceMotion();
      setBallReference();
      phase_speed_ = 0;
    }
    else if (current_node_->id == 6) {
      setChestStallMotion();
      phase_speed_ = 0;
    }
    else if (current_node_->id == 7 or current_node_->id == 8) {
      setChestMotion();
      setBallReference();
    }
    else if (current_node_->id == 11 or current_node_->id == 12) {
      setReferenceMotionHead();
      setBallReference();
      phase_speed_ = 0;
    }

    pTarget_.tail(36) = reference_.tail(36);

    //walker_->setState(reference_, gv_init_);
    //soccer_->setState(ball_reference_, ball_reference_vel_);

    walker_->setPdTarget(pTarget_, vTarget_);
    Eigen::VectorXd torque = Eigen::VectorXd::Zero(gvDim_);

    torque.segment(6, 28) = action.cast<double>() * 100.0;
    torque.segment(9, 3) *= 0.5;
    torque.segment(20, 4) *= 2.5;
    torque.segment(24, 3) *= 1.5;
    torque.segment(31, 3) *= 1.5;
    torque.segment(27, 4) *= 2.5;
    torque.segment(12, 8) *= 0.5;

    int num_contact = 0;
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      // walker_->getState(gc_, gv_);
      walker_->setGeneralizedForce(torque);
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
      for(auto& contact: soccer_->getContacts()) {
        num_contact++;
        if (phase_speed_ > 0.1 && (phase_ < max_phase_ / 2 - 3 * phase_speed_ || phase_ > max_phase_ / 2 + 3 * phase_speed_))
        {
          contact_terminal_flag_ = true;
          break;
        }
        soccer_->getState(ball_gc_, ball_gv_);
        if (ball_gc_[2] < 0.2) {
          contact_terminal_flag_ = true;
          break;
        }
        auto& pair_contact = world_->getObject(contact.getPairObjectIndex())->getContacts()[contact.getPairContactIndexInPairObject()];

        switch (current_node_->id) {
          case 9:
          case 10:
          case 11:
          case 12:
            if (walker_->getBodyIdx("neck") != pair_contact.getlocalBodyIndex()){
              contact_terminal_flag_ = true;
            }
            break;
          case 13:
          case 14:
            if (walker_->getBodyIdx("left_hip") != pair_contact.getlocalBodyIndex() 
          && walker_->getBodyIdx("left_knee") != pair_contact.getlocalBodyIndex())
              contact_terminal_flag_ = true;
            break;
          case 17:
          case 18:
            if (walker_->getBodyIdx("right_hip") != pair_contact.getlocalBodyIndex()
          && walker_->getBodyIdx("right_knee") != pair_contact.getlocalBodyIndex())
              contact_terminal_flag_ = true;
            break;
          case 0:
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
            if (walker_->getBodyIdx("left_ankle") != pair_contact.getlocalBodyIndex()
           && walker_->getBodyIdx("left_knee") != pair_contact.getlocalBodyIndex()) 
              contact_terminal_flag_ = true;
            break;
          case 6:
          case 7:
          case 8:
            if (walker_->getBodyIdx("chest") != pair_contact.getlocalBodyIndex() && walker_->getBodyIdx("neck") != pair_contact.getlocalBodyIndex()) 
                contact_terminal_flag_ = true;
              break;
          case 15:
          case 16:
          case 19:
          case 20:
          case 21:
          case 22:
            if (walker_->getBodyIdx("right_ankle") != pair_contact.getlocalBodyIndex()
            && walker_->getBodyIdx("right_knee") != pair_contact.getlocalBodyIndex())
              contact_terminal_flag_ = true;
            break;
          default:
            contact_terminal_flag_ = false;
        }
      }
    }

    if ((previous_node_->id == 4 || previous_node_->id == 5 || previous_node_->id == 6 || 
      previous_node_->id == 11 || previous_node_->id == 12 || previous_node_->id == 21 || previous_node_->id == 22) 
      && phase_speed_ < 0.1) {

      raisim::Vec<3> footPosition;
      if (previous_node_->id == 4 || previous_node_->id == 5) {
        auto footFrameIndex = walker_->getFrameIdxByName("left_ankle");
        walker_->getFramePosition(footFrameIndex, footPosition);
      }
      else if (previous_node_->id == 21 or previous_node_->id == 22) {
        auto footFrameIndex = walker_->getFrameIdxByName("right_ankle");
        walker_->getFramePosition(footFrameIndex, footPosition);
      }
      soccer_->getState(ball_gc_, ball_gv_);
      float foot_ball_distance = std::pow(footPosition[0] - ball_gc_[0], 2) + std::pow(footPosition[1] - ball_gc_[1], 2)
        + std::pow(footPosition[2] - ball_gc_[2], 2);

      if (foot_ball_distance > 0.09) {
        contact_terminal_flag_ = true;
      }
    }

    phase_ += phase_speed_;
    sim_step_ += 1;
    if ((phase_ > max_phase_ / 2 - phase_speed_ / 2 && phase_ < max_phase_ / 2 + phase_speed_ / 2) ||
      current_node_->id == 4 || current_node_->id == 5 || current_node_->id == 6 || current_node_->id == 11 || current_node_->id == 12
      || current_node_->id == 21 || current_node_->id == 22) {
      phase_ = max_phase_ / 2;
      if (previous_node_->id != -1) {
        update_previous_node();
      }
      update_current_node();
      previous_node_ = current_node_;
      current_node_ = next_node_;
      next_node_ = next_next_node_;
      next_next_node_ = next_node_->neighbour[select_next_node(next_node_)];

      if (current_node_->id == 0 or current_node_->id == 1 or current_node_->id == 7 or current_node_->id == 8 or 
        current_node_->id == 9 or current_node_->id == 10 or current_node_->id == 13 or current_node_->id == 14
        or current_node_->id == 15 or current_node_->id == 16 or current_node_->id == 17 or current_node_->id == 18)
        calculate_phase_speed(desired_max_height_);
      else if (current_node_->id == 2 or current_node_->id == 3 or current_node_->id == 19 or current_node_->id == 20)
        phase_speed_ = max_phase_ / 30;
      else if (current_node_->id == 4 or current_node_->id == 5 or current_node_->id == 11 or current_node_->id == 12 
        or current_node_->id == 21 or current_node_->id == 22)
        phase_speed_ = 0;
      task_vector_.setZero();
      task_vector_[current_node_->id] = 1;
      next_task_vector_.setZero();
      next_task_vector_[next_node_->id] = 1;
      next_next_task_vector_.setZero();
      next_next_task_vector_[next_next_node_->id] = 1;
    }
    else if (phase_ >= max_phase_){
      phase_ = 0;

      update_previous_node();
      update_current_node();

      previous_node_ = current_node_;
      current_node_ = next_node_;
      next_node_ = next_next_node_;
      next_next_node_ = next_node_->neighbour[select_next_node(next_node_)];


      // update contact height
      task_vector_.setZero();
      task_vector_[current_node_->id] = 1;
      next_task_vector_.setZero();
      next_task_vector_[next_node_->id] = 1;
      next_next_task_vector_.setZero();
      next_next_task_vector_[next_next_node_->id] = 1;
      
      // update phase speed
      if (current_node_->id == 0 or current_node_->id == 1 or current_node_->id == 13 or current_node_->id == 14
        or current_node_->id == 15 or current_node_->id == 16 or current_node_->id == 17 or current_node_->id == 18)
        calculate_phase_speed(desired_max_height_);
      else if (current_node_->id == 7 or current_node_->id == 8 or current_node_->id == 9 or current_node_->id == 10)
        calculate_phase_speed(desired_max_height_);
      else if (current_node_->id == 2 or current_node_->id == 3 or current_node_->id == 19 or current_node_->id == 20)
        phase_speed_ = max_phase_ / 30;

      //update next desired max height
      switch (next_next_node_->id) {
        case 9:
        case 10:
        case 11:
        case 12:
          desired_max_height_ = 2.0;
          break;
        case 0:
        case 1:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
           if (current_node_->id == 7 or current_node_->id == 8 or current_node_->id == 9 or current_node_->id == 10 
            or current_node_->id == 11 or current_node_->id == 12)
            desired_max_height_ = 2.0;
           else
            desired_max_height_ = 1.4;
           break;
        case 7:
        case 8:
          desired_max_height_ = 2.0;
          break;
        default:
          desired_max_height_ = 1.4;
      }
    }

    updateObservation();
    computeReward();
    double current_reward = rewards_.sum() * 0.5 + (rewards_["position"] + rewards_["orientation"] + rewards_["joint"]) * rewards_["ball position"] * 0.0;
    total_reward_ +=  current_reward;

    return  current_reward;
  }

  void computeReward() {
    float joint_reward = 0, position_reward = 0, orientation_reward = 0;
    
    //compute joint reward
    raisim::Vec<4> quat, quat2, quat_error;
    raisim::Mat<3,3> rot, rot2, rot_error;
    
    for (int j = 0; j < 12; j++) {
      if (j == 0 or j == 1 or j == 2 or j == 4 or j == 6 or j == 8 or j == 9 or j == 11) {
        quat[0] = gc_[joint_start_index[j]]; quat[1] = gc_[joint_start_index[j]+1]; 
        quat[2] = gc_[joint_start_index[j]+2]; quat[3] = gc_[joint_start_index[j]+3];
        quat2[0] = reference_[joint_start_index[j]]; quat2[1] = reference_[joint_start_index[j]+1]; 
        quat2[2] = reference_[joint_start_index[j]+2]; quat2[3] = reference_[joint_start_index[j]+3];
        raisim::quatToRotMat(quat, rot);
        raisim::quatToRotMat(quat2, rot2);
        raisim::mattransposematmul(rot, rot2, rot_error);
        raisim::rotMatToQuat(rot_error, quat_error);
        joint_reward += 1*(std::pow(quat_error[1], 2) + std::pow(quat_error[2], 2) + std::pow(quat_error[3], 2));
      }
      else {
        joint_reward += std::pow(gc_[joint_start_index[j]] - reference_[joint_start_index[j]], 2);
      }
    }

    position_reward += 1.0 * std::pow(gv_[0]-speed, 2) + std::pow(gv_[1]-0, 2) + std::pow(gc_[2]-reference_[2], 2);
    
    orientation_reward += (std::pow(gc_[4]-reference_[4], 2)) + (std::pow(gc_[5]-reference_[5], 2)) + (std::pow(gc_[6]-reference_[6], 2));

    float ball_position_reward = 0;
    ball_position_reward += std::pow(ball_reference_[0] - ball_gc_[0] + gc_[0], 2) + 
      std::pow(ball_reference_[1] - ball_gc_[1] + gc_[1], 2) + 5 * std::pow(ball_reference_[2] - ball_gc_[2], 2);

    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-2 * orientation_reward));
    rewards_.record("joint", std::exp(-3*joint_reward));
    rewards_.record("ball position", std::exp(-ball_position_reward));
  }

  void updateObservation() {
    soccer_->getState(ball_gc_, ball_gv_);
    walker_->getState(gc_, gv_);

    obDouble_ << gc_[2], /// body height
        (gc_[3] - 0.707) * 4, (gc_[4] - 0.707) * 4, gc_[5] * 4, gc_[6] * 4,/// body orientation
        gc_[7], gc_[8] * 2, gc_[9] * 2, gc_[10] * 2, //chest
        gc_[11], gc_[12] * 2, gc_[13] * 2, gc_[14] * 2, //chest
        gc_.tail(28), /// joint angles
        gv_[0], gv_[1], gv_[2], gv_[3], gv_[4], gv_[5],/// body linear&angular velocity
        gv_.tail(28) / 10.0, /// joint velocity
        speed, //speed
        ball_gc_[0] - gc_[0], ball_gc_[1] - gc_[1], ball_gc_[2] - gc_[2],
        ball_gv_[0]/10.0, ball_gv_[1]/10.0, ball_gv_[2]/10.0, ball_gv_[3]/10.0, ball_gv_[4]/10.0, ball_gv_[5]/10.0,
        phase_speed_ / 10.0, //next_phase_speed_ / 10.0, 
        std::cos(phase_ * 3.1415 * 2 / max_phase_), std::sin(phase_ * 3.1415 * 2 / max_phase_),// phase;
        task_vector_.tail(num_task_), next_task_vector_.tail(num_task_), next_next_task_vector_.tail(num_task_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  void getState(Eigen::Ref<EigenVec> ob) final {
    stateDouble_ << gc_.tail(gcDim_), ball_gc_.tail(7), reference_.tail(gcDim_), ball_reference_.tail(7);
    ob = stateDouble_.cast<float>();
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }

  float get_total_reward() {
    return float(total_reward_);
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_) * 0.0f;

    raisim::Vec<4> quat, quat2, quat_error;
    raisim::Mat<3,3> rot, rot2, rot_error;
    quat[0] = gc_[3]; quat[1] = gc_[4]; 
    quat[2] = gc_[5]; quat[3] = gc_[6];
    quat2[0] = 0.707; quat2[1] = 0.707; 
    quat2[2] = 0; quat2[3] = 0;
    raisim::quatToRotMat(quat, rot);
    raisim::quatToRotMat(quat2, rot2);
    raisim::mattransposematmul(rot, rot2, rot_error);
    raisim::rotMatToQuat(rot_error, quat_error);

    if ((std::pow(quat_error[1], 2) + std::pow(quat_error[2], 2) + std::pow(quat_error[3], 2)) > 0.06) {
      return true;
    }
    if (std::abs(gc_[2]) < 0.6) {
      return true;
    }
    if (std::abs(ball_gc_[0] - gc_[0] - ball_reference_[0]) > 1.0 || 
      std::abs(ball_gc_[1] - gc_[1] - ball_reference_[1]) > 0.5 || ball_gc_[2] < 0.2) {
      return true;
    }
    if (contact_terminal_flag_) {
      return true;
    }

    auto footFrameIndex = walker_->getFrameIdxByName("left_ankle"); // the URDF has a joint named "foot_joint"
    raisim::Vec<3> footPosition;
    walker_->getFramePosition(footFrameIndex, footPosition);

    if ((current_node_->id == 2 || current_node_->id == 3) && (phase_ < max_phase_ - phase_speed_ && phase_ > phase_speed_)) {
      float ball_foot_angle_tan = atan2(footPosition[2] - ball_gc_[2], footPosition[1] - ball_gc_[1]);
      int index = int(phase_ - phase_speed_) % int(max_phase_) / max_phase_ * 30;
      if (
        (std::pow(cos(ball_foot_angle_tan) - around_the_world_cos_constraint[index], 2)+
          std::pow(sin(ball_foot_angle_tan) - around_the_world_sin_constraint[index], 2) > 1)){
        return true;
      }
    }

    auto rightfootFrameIndex = walker_->getFrameIdxByName("right_ankle"); // the URDF has a joint named "foot_joint"
    walker_->getFramePosition(rightfootFrameIndex, footPosition);

    if ((current_node_->id == 19 || current_node_->id == 20) && (phase_ < max_phase_ - phase_speed_ && phase_ > phase_speed_)) {
      float ball_foot_angle_tan = atan2(footPosition[2] - ball_gc_[2], footPosition[1] - ball_gc_[1]);
      int index = int(phase_ - phase_speed_) % int(max_phase_) / max_phase_ * 30;
      if (
        (std::pow(cos(ball_foot_angle_tan) + around_the_world_cos_constraint[index], 2)+
          std::pow(sin(ball_foot_angle_tan) - around_the_world_sin_constraint[index], 2) > 1)){
        return true;
      }
    }

    if (current_node_->id == 11 or current_node_->id == 12) {
      if (std::abs(gc_[26]) > 0.01 or std::abs(gc_[27]) > 0.01 or std::abs(gc_[28]) > 0.01 
        or std::abs(gc_[35]) > 0.01 or std::abs(gc_[36]) > 0.01 or std::abs(gc_[37]) > 0.01) 
        return true;
    }

    return false;
  }

  void setReferenceMotion() {
    reference_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;
    reference_[0] = speed * 0.02 * sim_step_;
    reference_[2] = 0.9;
    //kicking with one foot
    reference_[25 + 9 * ((current_node_->id == 0 || current_node_->id==1 || current_node_->id == 4 || current_node_->id == 5
      || current_node_->id == 13 || current_node_->id == 14) % 2) + 4] 
      = -1.57 * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);

    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    euler[0] = 0;
    euler[1] = 0;
    euler[2] = 1.57 * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(25 + 9 * ((current_node_->id == 0 || current_node_->id==1 || current_node_->id == 4 || 
      current_node_->id == 5 || current_node_->id == 13 || current_node_->id == 14) % 2), 4) 
      << quat[0], quat[1], quat[2], quat[3];

    reference_[19] = 1.57;
    reference_[24] = 1.57;

    euler[0] = 0.7;
    euler[1] = 0;
    euler[2] = 0;
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(15, 4) << quat[0], -quat[1], quat[2], quat[3];
    reference_.segment(20, 4) << quat[0], quat[1], quat[2], quat[3];
  }

  void setReferenceMotionHead() {
    reference_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;

    reference_[0] = speed * 0.02 * sim_step_;
    reference_[2] = 0.9;
    float final_angle = 0.7;
    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    if (phase_ < max_phase_ / 2) {
      euler[0] = 0;
      euler[1] = 0;
      euler[2] = final_angle * std::sin(phase_ / max_phase_ * 3.1415 * 2);
      raisim::eulerVecToQuat(euler, quat);
      reference_.segment(30, 4) << quat[0], quat[1], quat[2], quat[3];
      reference_.segment(39, 4) << quat[0], quat[1], quat[2], quat[3];
      reference_[29] = -final_angle * std::sin(phase_ / max_phase_ * 3.1415 * 2);
      reference_[38] = -final_angle * std::sin(phase_ / max_phase_ * 3.1415 * 2);
    }
    reference_[2] -= (0.5 - 0.5 * std::cos(reference_[29]));
    reference_[0] += 0.5 * std::sin(-reference_[29]);

    reference_[19] = 1.57;
    reference_[24] = 1.57;
    euler[0] = 0.7;
    euler[1] = 0;
    euler[2] = 0;
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(15, 4) << quat[0], -quat[1], quat[2], quat[3];
    reference_.segment(20, 4) << quat[0], quat[1], quat[2], quat[3];
  }

  void setBallReference() {
    ball_reference_ *= 0;
    ball_reference_[0] = ball_gc_init_[0];
    ball_reference_[3] = 1;
    float init_vel = g_ * max_phase_ * control_dt_ / 2 / phase_speed_;
    
    float init_height = 0;

    if (current_node_->id == 9 or current_node_->id == 10 or current_node_->id == 11 or current_node_->id == 12) {
      init_height = 1.6;
      //ball_reference_[0] = 0.0;
    }
    else if (current_node_->id == 13 || current_node_->id == 14 || current_node_->id == 17 or current_node_->id == 18) {
      init_height = 1.0;  // 1 for knee juggling 0.5 for foot
      //ball_reference_[0] = 0.4;  //0.5 for foot 0.4 for knee
    }
    else if (current_node_->id == 7 or current_node_->id == 8) {
      init_height = 1.3;
    }
    else {
      init_height = 0.5;  // 1 for knee juggling 0.5 for foot
      //ball_reference_[0] = 0.65;  //0.5 for foot 0.4 for knee
    }

    if (phase_ <= max_phase_ /2) {
      if (current_node_->id == 9 or current_node_->id == 10 or current_node_->id == 11 or current_node_->id == 12) {
        ball_reference_[0] = 0.0;
        ball_reference_[1] = 0.0;
      }
      else if (current_node_->id == 13 || current_node_->id == 14 || current_node_->id == 17 || current_node_->id == 18) {
        ball_reference_[0] = 0.4;  //0.5 for foot 0.4 for knee
        ball_reference_[1] = 0.1 * (((current_node_->id == 13 || current_node_->id== 14) % 2) * 2 - 1);
      }
      else if (current_node_->id == 7 or current_node_->id == 8) {
        ball_reference_[0] = 0.25;
        ball_reference_[1] = 0;
      }
      else {
        ball_reference_[0] = 0.6;  //0.5 for foot 0.4 for knee
        ball_reference_[1] = 0.1 * (((current_node_->id == 0 || current_node_->id==1 || current_node_->id == 4 || current_node_->id == 5) % 2) * 2 - 1);
      }
    }
    else {
      if (next_node_->id == 9 or next_node_->id == 10 or next_node_->id == 11 or next_node_->id == 12) {
        ball_reference_[0] = 0.0;
        ball_reference_[1] = 0.0;
      }
      else if (next_node_->id == 13 || next_node_->id == 14 || current_node_->id == 17 || current_node_->id == 18) {
        ball_reference_[0] = 0.4;  //0.5 for foot 0.4 for knee
        ball_reference_[1] = 0.1 * (((next_node_->id == 13 || next_node_->id== 14) % 2) * 2 - 1);
      }
      else if (next_node_->id == 7 or next_node_->id == 8) {
        ball_reference_[0] = 0.25;
        ball_reference_[1] = 0;
      }
      else {
        ball_reference_[0] = 0.6;  //0.5 for foot 0.4 for knee
        ball_reference_[1] = 0.1 * (((next_node_->id == 0 || next_node_->id==1 || next_node_->id == 4 || next_node_->id == 5) % 2) * 2 - 1);
      }
    }
    
    float max_height = 0.5 * g_ * std::pow(max_phase_ / 2 * control_dt_  / phase_speed_ , 2)+ init_height;
    if (current_node_->id == 4 or current_node_->id == 5 or current_node_->id == 21 or current_node_->id == 22) {
      ball_reference_[2] = 0.5;
    }
    else if (current_node_->id == 11 or current_node_->id == 12) {
      ball_reference_[2] = 1.6;
    }
    else if (phase_ > max_phase_ / 2) {
      float cur_vel = init_vel - g_ * (phase_ - max_phase_ / 2) * control_dt_ / phase_speed_;
      ball_reference_[2] = (init_vel + cur_vel) / 2.0 * control_dt_ * (phase_ - max_phase_/2) / phase_speed_ + init_height;
      ball_reference_vel_[2] = cur_vel;
    }
    else {
      ball_reference_[2] = max_height - 0.5 * g_ * std::pow((phase_)*control_dt_ / phase_speed_, 2);
      ball_reference_vel_[2] = -g_ * (phase_)*control_dt_ / phase_speed_;
    }
  }

  void calculate_phase_speed(float max_height) {
    float final_height = 0.0;
    if (current_node_->id == 9 or current_node_->id == 10) final_height = 1.6;
    else if (current_node_->id == 3 || current_node_->id == 4 || current_node_->id == 21 || current_node_->id == 22) final_height = 1.0;
    else if (current_node_->id == 7 || current_node_->id == 8) final_height = 1.31;
    else if (current_node_->id == 13 || current_node_->id == 14 || current_node_->id == 17 || current_node_->id == 18) final_height = 1.0;
    else final_height = 0.5;
    float time_to_fall = std::sqrt((max_height - final_height) * 2 / g_);
    phase_speed_ = control_dt_ * max_phase_ / 2.0 / time_to_fall;
  }

  void read_around_the_world() {
    // data 030004_001_20_T_ST_0100_2_JM_Player2_Standard/Input.txt frame 414, num_frame 30
    std::ifstream infile("around_the_world.txt");
    float data;
    int i = 0, j = 0;
    while (infile >> data) {
      if (i < 43) {
        around_the_world_reference_.coeffRef(j, i) = data;
      }
      else {
        around_the_world_ball_reference_.coeffRef(j, i-43) = data;
      }
      i++;
      if (i >= 46) {
        i = 0;
        j++;
      }
    }
  }

  void read_around_the_world_right() {
    // data 030004_001_20_T_ST_0100_2_JM_Player2_Standard/Input.txt frame 414, num_frame 30
    std::ifstream infile("around_the_world_right.txt");
    float data;
    int i = 0, j = 0;
    while (infile >> data) {
      if (i < 43) {
        around_the_world_right_reference_.coeffRef(j, i) = data;
      }
      else {
        around_the_world_ball_right_reference_.coeffRef(j, i-43) = data;
      }
      i++;
      if (i >= 46) {
        i = 0;
        j++;
      }
    }
  }

  void setReferenceMotionAroundTheWorld() {
    int index = phase_ / max_phase_ * 30;
    reference_.segment(0, 43) = around_the_world_reference_.row(index);
    ball_reference_.segment(0, 3) = around_the_world_ball_reference_.row(index);
    ball_reference_.segment(3, 4) << 1, 0, 0, 0;
    reference_[2] += 0.05;
    ball_reference_[2] += 0.05;
  }

  void setReferenceMotionAroundTheWorldRight() {
    int index = phase_ / max_phase_ * 30;
    reference_.segment(0, 43) = around_the_world_right_reference_.row(index);
    reference_.segment(7, 8) = around_the_world_reference_.row(index).segment(7, 8);
    reference_.segment(15, 4) = around_the_world_reference_.row(index).segment(20, 4);
    reference_.segment(20, 4) = around_the_world_reference_.row(index).segment(15, 4);
    reference_[8] *= -1; reference_[9] *= -1; reference_[12] *= -1; reference_[13] *= -1;
    reference_[16] *= -1; reference_[17] *= -1; reference_[21] *= -1; reference_[22] *= -1;
    ball_reference_.segment(0, 3) = around_the_world_ball_right_reference_.row(index);
    ball_reference_.segment(3, 4) << 1, 0, 0, 0;
    reference_[2] += 0.05;
    ball_reference_[2] += 0.05;
  }

  void setFootStallMotion() {
    reference_ << 0.00000000e+00,0.00000000e+00,9.00000000e-01,7.07106781e-01
      ,7.07106781e-01,1.32379300e-17,-3.63717181e-17,9.89860000e-01
      ,-2.95500000e-02,1.38360000e-01,1.26500000e-02,9.70090000e-01
      ,2.16900000e-02,2.40380000e-01,2.58600000e-02,9.73580000e-01
      ,-1.44300000e-01,5.88500000e-02,-1.66870000e-01,6.52466861e-01
      ,9.76880000e-01,1.59560000e-01,-9.24300000e-02,-1.08170000e-01
      ,9.27183706e-01,9.90510000e-01,-3.70200000e-02,-1.28370000e-01
      ,3.23500000e-02,-4.73398918e-01,9.93730000e-01,-5.50100000e-02
      ,-9.73000000e-03,-9.68400000e-02,9.78940000e-01,-3.43600000e-02
      ,-3.46700000e-02,1.98220000e-01,-3.18718265e-01,9.93800000e-01
      ,3.68900000e-02,-4.09900000e-02,-9.65500000e-02;

    ball_reference_ << 0.3441501, 0.00392874, 0.24349171, 1, 0, 0, 0;
  }

  void setChestMotion() {

    reference_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;

    reference_[0] = speed * 0.02 * sim_step_;
    reference_[2] = 0.9;
    float final_angle = 0.46;
    raisim::Vec<4> quat;
    raisim::Vec<3> euler;
    euler[0] = 0;
    euler[1] = 0;
    euler[2] = final_angle * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(30, 4) << quat[0], quat[1], quat[2], quat[3];
    reference_.segment(39, 4) << quat[0], quat[1], quat[2], quat[3];
    reference_[29] = -final_angle;
    reference_[38] = -final_angle;
    reference_[2] -= (0.5 - 0.5 * std::cos(reference_[29]));
    reference_[0] += 0.5 * std::sin(-reference_[29]);

    float waist_angle = 0.8;
    euler[0] = 0;
    euler[1] = 0;
    euler[2] = waist_angle * std::sin(phase_ * 1.0 / max_phase_ * 3.1415);
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(7, 4) << quat[0], quat[1], quat[2], quat[3];

    reference_[19] = 1.57;
    reference_[24] = 1.57;
    euler[0] = 0.7;
    euler[1] = 0;
    euler[2] = 0;
    raisim::eulerVecToQuat(euler, quat);
    reference_.segment(15, 4) << quat[0], -quat[1], quat[2], quat[3];
    reference_.segment(20, 4) << quat[0], quat[1], quat[2], quat[3];

    ball_reference_ << 0.25, 0.02254292, 1.3064698, 1, 0, 0, 0;
  }

  void setChestStallMotion() {

     reference_ << 0, 0, 1.707, 0.707, 0.707, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 
      0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0;

      reference_[0] = speed * 0.02 * sim_step_;
      reference_[2] = 0.9;
      float final_angle = 0.46;
      raisim::Vec<4> quat;
      raisim::Vec<3> euler;
      euler[0] = 0;
      euler[1] = 0;
      euler[2] = final_angle;
      raisim::eulerVecToQuat(euler, quat);
      reference_.segment(30, 4) << quat[0], quat[1], quat[2], quat[3];
      reference_.segment(39, 4) << quat[0], quat[1], quat[2], quat[3];
      reference_[29] = -final_angle;
      reference_[38] = -final_angle;
      reference_[2] -= (0.5 - 0.5 * std::cos(reference_[29]));
      reference_[0] += 0.5 * std::sin(-reference_[29]);

      float waist_angle = 0.8;
      euler[0] = 0;
      euler[1] = 0;
      euler[2] = waist_angle;
      raisim::eulerVecToQuat(euler, quat);
      reference_.segment(7, 4) << quat[0], quat[1], quat[2], quat[3];

      reference_[19] = 1.57;
      reference_[24] = 1.57;
      euler[0] = 0.7;
      euler[1] = 0;
      euler[2] = 0;
      raisim::eulerVecToQuat(euler, quat);
      reference_.segment(15, 4) << quat[0], -quat[1], quat[2], quat[3];
      reference_.segment(20, 4) << quat[0], quat[1], quat[2], quat[3];

      ball_reference_ << 0.25, 0.02254292, 1.3064698, 1, 0, 0, 0;
    }


 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  bool flip_obs_ = false;
  raisim::ArticulatedSystem* walker_;
  raisim::ArticulatedSystem* soccer_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd ball_gc_init_, ball_gv_init_, ball_gc_, ball_gv_;
  Eigen::VectorXd reference_;
  Eigen::VectorXd ball_reference_, ball_reference_vel_;
  Eigen::MatrixXd around_the_world_reference_;
  Eigen::MatrixXd around_the_world_ball_reference_;
  Eigen::MatrixXd around_the_world_right_reference_;
  Eigen::MatrixXd around_the_world_ball_right_reference_;
  // Eigen::VectorXd foot_stall_reference_, foot_stall_ball_reference_;
  // Eigen::VectorXd chest_stall_reference_, chest_stall_ball_reference_;
  float phase_ = 0; //use to be int
  float max_phase_ = 150;
  float phase_speed_ = 5;
  float next_phase_speed_ = 0;
  double desired_max_height_ = 1.4;
  int sim_step_ = 0;
  int max_sim_step_ = 1000;
  int juggle_leg_ = 0;
  int use_knee_ = 1;
  int use_head_ = 0;
  int next_use_knee_ = 1;
  int next_use_head_ = 0;
  int use_around_the_world_ = 0;
  int next_use_around_the_world_ = 0;
  bool contact_terminal_flag_ = false;
  double total_reward_ = 0;
  double terminalRewardCoeff_ = 0.;
  double speed = 0.0;
  int joint_start_index[12] = {7, 11, 15, 19, 20, 24, 25, 29, 30, 34, 38, 39};
  float around_the_world_cos_constraint[30] = {0.803, 0.690, 0.588, 0.395, 0.315, 0.259, 0.196, 0.155, 0.057, 0,
                                                -0.048, -0.118, -0.201, -0.532, -0.648, -0.836, -0.647, -0.667, -0.805, -0.887,
                                                -0.945, -0.992, -0.997, -0.839, -0.659, -0.429, -0.172, 0.104, 0.614, 0.833};
  float around_the_world_sin_constraint[30] = {-0.595, -0.723, -0.809, -0.919, -0.949, -0.966, -0.981, -0.988, -0.998, -1,
                                                -0.999, -0.993, -0.980, -0.847, -0.761, -0.549, -0.763, -0.745, -0.593, -0.462,
                                                -0.327, -0.127, 0.078, 0.543, 0.752, 0.903, 0.985, 0.995, 0.789, 0.553};
  float around_the_world_cos_constraint_right[30] = {0.803, 0.690, 0.588, 0.395, 0.315, 0.259, 0.196, 0.155, 0.057, 0,
                                                -0.048, -0.118, -0.201, -0.532, -0.648, -0.836, -0.647, -0.667, -0.805, -0.887,
                                                -0.945, -0.992, -0.997, -0.839, -0.659, -0.429, -0.172, 0.104, 0.614, 0.833};
  float around_the_world_sin_constraint_right[30] = {0.595, 0.723, 0.809, 0.919, 0.949, 0.966, 0.981, 0.988, 0.998, 1,
                                                0.999, 0.993, 0.980, 0.847, 0.761, 0.549, 0.763, 0.745, 0.593, 0.462,
                                                0.327, 0.127, -0.078, -0.543, -0.752, -0.903, -0.985, -0.995, -0.789, -0.553};
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::VectorXd stateDouble_;
  // Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  // Eigen::Vector3d leftHandVel_, rightHandVel_;
  std::set<size_t> footIndices_;
  int mode_ = 0;
  float g_ = 9.81;
  int num_task_ = 23;
  Eigen::VectorXd task_vector_, next_task_vector_, next_next_task_vector_;
  int task_index_ = 0;
  int next_task_index_ = 0;

  // control graph
  Node* foot_juggle_down_node_ = new Node();
  Node* foot_juggle_up_node_ = new Node();
  Node* around_the_world_up_node_ = new Node();
  Node* around_the_world_down_node_ = new Node();

  Node* foot_stall_enter_node_ = new Node();
  Node* foot_stall_exit_node_ = new Node();
  Node* chest_stall_node_ = new Node();
  Node* chest_juggle_up_node_ = new Node();
  Node* chest_juggle_down_node_ = new Node();
  Node* head_juggle_up_node_ = new Node();
  Node* head_juggle_down_node_ = new Node();
  Node* head_stall_node_ = new Node();
  Node* head_stall_exit_node_ = new Node();
  Node* knee_juggle_down_node_ = new Node();
  Node* knee_juggle_up_node_ = new Node();
  Node* right_foot_juggle_down_node_ = new Node();
  Node* right_foot_juggle_up_node_ = new Node();
  Node* right_knee_juggle_down_node_ = new Node();
  Node* right_knee_juggle_up_node_ = new Node();
  Node* right_around_the_world_down_node_ = new Node();
  Node* right_around_the_world_up_node_ = new Node();
  Node* right_foot_stall_node_ = new Node();
  Node* right_foot_stall_exit_node_ = new Node();

  Node * current_node_ = new Node();
  Node * next_node_ = new Node();
  Node * next_next_node_ = new Node();
  Node * previous_node_ = new Node();
  Node *psuduo_node_ = new Node();
  std::vector<Node*> initial_node;
  int previous_choice_ = -1;
  int previous_previous_choice_ = -1;
  int previous_previous_previous_choice_ = -1;
  int foot_stall_counter = 0;
};
}