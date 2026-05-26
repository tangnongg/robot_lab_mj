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
    // const auto cfg = params["command_name"].as<std::string>();
    const auto cfg = env->cfg["commands"]["base_pose"]["ranges"];
    // roll, pitch, yaw, base_height
    obs[0] = std::clamp(joystick->lx(), cfg["roll"][0].as<float>(), cfg["roll"][1].as<float>());
    obs[1] = std::clamp(joystick->ly(), cfg["pitch"][0].as<float>(), cfg["pitch"][1].as<float>());
    obs[2] = std::clamp(joystick->rx(), cfg["yaw"][0].as<float>(), cfg["yaw"][1].as<float>());
    obs[3] = std::clamp(joystick->ry(), cfg["base_height"][0].as<float>(), cfg["base_height"][1].as<float>());
    std::cout << "command obs: " << obs[0] << " " << obs[1] << " " << obs[2] << " " << obs[3] << std::endl;
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