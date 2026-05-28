#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

// like in deploy/include/isaaclab/envs/mdp/observations/observations.h, 
// define REGISTER_OBSERVATION(name)

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(ryp_bh_command)
{
    std::vector<float> obs(4);
    auto joystick = env->robot->data.joystick;
    // const auto cfg = params["command_name"].as<std::string>(); # avaliable way to get params
    const auto cfg = env->cfg["commands"]["base_pose"]["offset_ranges"];
    // roll, pitch, yaw, base_height
    if (joystick->lx() < 0) {
        obs[0] = -joystick->lx() * cfg["roll_offset"][0].as<float>();
    }
    else {
        obs[0] = joystick->lx() * cfg["roll_offset"][1].as<float>();
    }
    if (joystick->ly() < 0) {
        obs[1] = -joystick->ly() * cfg["pitch_offset"][0].as<float>();
    }
    else {
        obs[1] = joystick->ly() * cfg["pitch_offset"][1].as<float>();
    }
    if (joystick->rx() < 0) {
        obs[2] = -joystick->rx() * cfg["yaw_offset"][0].as<float>();
    }
    else {
        obs[2] = joystick->rx() * cfg["yaw_offset"][1].as<float>();
    }
    if (joystick->ry() < 0) {
        obs[3] = -joystick->ry() * cfg["base_height_offset"][0].as<float>() + 0.58;
    }
    else {
        obs[3] = joystick->ry() * cfg["base_height_offset"][1].as<float>() + 0.58;
    }
    // std::cout << "ryp_bh_command: " << obs[0] << " " << obs[1] << " " << obs[2] << " " << obs[3] << std::endl;
    return obs;
}

}
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}