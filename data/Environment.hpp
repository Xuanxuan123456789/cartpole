#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
using namespace std;
#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)
namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {
 public:
  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    // 创建世界
    world_ = std::make_unique<raisim::World>();

    // 添加倒立摆模型
    cartpole_ = world_->addArticulatedSystem(resourceDir + "/cartpole.urdf");
    cartpole_->setName("cartpole");
    cartpole_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    world_->addGround();
    // 获取模型参数
    gcDim_ = cartpole_->getGeneralizedCoordinateDim();
    gvDim_ = cartpole_->getDOF();
    nJoints_ = 2;
    // 初始化容器
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    cartpole_->setGeneralizedForce(Eigen::VectorXd::Zero(nJoints_));

    // 观测和动作空间定义
    obDim_ = 4;  // x, theta, x_dot, theta_dot
    actionDim_ = 1;

    // 动作和观测的归一化参数
    actionMean_.setZero(actionDim_);
    actionStd_.setConstant(actionDim_);
    obMean_.setZero(obDim_);
    obStd_.setZero(obDim_);
    actionMean_ = gc_init_.tail(actionDim_);
    actionStd_.setConstant(0.6);
    obMean_.setZero();
    obStd_ << 3, 2 * M_PI, 0.1, 10.0;

    // 奖励系数
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    // 可视化设置
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(cartpole_);
    }
  }

  void init() final { }

  void reset() final {
    cartpole_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  
  float step(const Eigen::Ref<EigenVec>& action) final {
    // 动作缩放
    actionScaled_ = action.cast<double>().cwiseProduct(actionStd_) + actionMean_;
    Eigen::Vector2d gf; 
    gf.setZero();
    gf.head(1) = actionScaled_;
    cartpole_->setGeneralizedForce(gf);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    updateObservation();

    rewards_.record("forceReward", cartpole_->getGeneralizedForce().squaredNorm());
    rewards_.record("survival", 1.0);
    return rewards_.sum();
  }
  
  
  void updateObservation() {
    cartpole_->getState(gc_, gv_);
    obDouble_.setZero(obDim_); obScaled_.setZero(obDim_);
    obDouble_ << gc_,gv_;
    obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);
  
  }
  
  void observe(Eigen::Ref<EigenVec> ob) final {
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    if (rad2deg(std::abs(obDouble_[1])) > 50.0 ||  // 杆角度超过50度
        std::abs(obDouble_[0]) > 2.0) {           // 小车位置超过2米
      return true;
    }
    terminalReward = 0.f;
    return false;
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  double reward_;
  bool visualizable_=false;
  raisim::ArticulatedSystem* cartpole_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, actionScaled_, torque_;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::unique_ptr<raisim::RaisimServer> server_;
  int visualizationCounter_=0;
  double forceRewardCoeff_ = 0., forceReward_ = 0.;
  double terminalRewardCoeff_ = -10.;
};

}