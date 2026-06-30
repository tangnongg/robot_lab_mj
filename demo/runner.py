"""Interactive SafeFall demo — full joystick control.

Default state: robot stands still (no auto‑walk).
Joystick left stick → velocity commands (vx, vy, yaw).
Button A          → random push (may cause fall).
Button B          → stand‑up (HoST policy).
Auto‑switch       → fall predictor detects → SafeFall policy.
Manual switch     → after landing, press B to stand up.

Joystick mapping (confirmed on this controller):
  Axis 0  — left stick horizontal    → vy (lateral, m/s)  ※ inverted below
  Axis 1  — left stick vertical      → vx (forward/back, m/s)
  Axis 3  — right stick horizontal   → yaw (turn rate, rad/s)
  Button 0 — A / Cross               → random push
  Button 1 — B / Circle              → stand‑up (HoST)
  Button 2 — X / Square              → switch to loco policy
  Button 6 — Select / Back           → reset environment
"""

from __future__ import annotations

import time
from typing import Callable

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from .joystick import JoystickReader


# ── State transfer ─────────────────────────────────────────────


def capture_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Snapshot robot qpos / qvel for transfer to another env."""
    a = env.scene["robot"].data
    return {
        "root_state": a.default_root_state[0:1].clone(),
        "joint_pos": a.joint_pos[0:1].clone(),
        "joint_vel": a.joint_vel[0:1].clone(),
    }


def restore_state(env: ManagerBasedRlEnv, state: dict[str, torch.Tensor]) -> None:
    """Write a state snapshot into a single‑env simulation."""
    a = env.scene["robot"]
    root = state["root_state"].to(env.device)
    jpos = state["joint_pos"].to(env.device)
    jvel = state["joint_vel"].to(env.device)
    a.write_root_state_to_sim(root, env_ids=torch.tensor([0]))
    a.write_joint_state_to_sim(jpos, jvel, env_ids=torch.tensor([0]))


# ── Viewer helpers ─────────────────────────────────────────────


def _sync(env: ManagerBasedRlEnv, viewer) -> None:
    sd = env.sim.data
    md = env.sim.mj_data
    md.qpos[:] = sd.qpos[0].cpu().numpy()
    md.qvel[:] = sd.qvel[0].cpu().numpy()
    md.xfrc_applied[:] = sd.xfrc_applied[0].cpu().numpy()
    mujoco.mj_forward(env.sim.mj_model, md)
    viewer.sync()


# ── Push generator ─────────────────────────────────────────────

def _random_push(rng: np.random.Generator) -> tuple[float, float]:
    """Random push strong enough to reliably destabilise G1.

    Uses xfrc_applied (force in N) directly on the torso body
    instead of velocity perturbation — more physical and
    guaranteed to trigger a fall regardless of controller state.
    """
    fx = float(rng.uniform(-500, 500))  # N — forward/backward
    fy = float(rng.uniform(-300, 300))  # N — lateral
    return fx, fy


def _apply_push(asset, fx: float, fy: float, device: str) -> None:
    """Apply a direct force impulse to the torso body."""
    forces = torch.zeros(1, asset.num_bodies, 3, device=device)
    forces[0, 0, 0] = fx  # torso body 0
    forces[0, 0, 1] = fy
    asset.write_external_wrench_to_sim(
        forces, torch.zeros_like(forces), env_ids=torch.tensor([0]))


# ── Joystick → velocity command ────────────────────────────────

# Scale factors: joystick [-1,1] → velocity command range.
_VEL_SCALE_X = 1.0   # forward speed at full stick
_VEL_SCALE_Y = 0.5   # lateral speed
_VEL_SCALE_YAW = 1.5  # yaw rate rad/s


def _js_to_cmd(js: JoystickReader, device: str) -> torch.Tensor:
    """Left stick vertical → vx, left stick horizontal → vy, right stick horizontal → yaw."""
    vx = js.axis(1, 0.0) * _VEL_SCALE_X * -1     # axis 1 → forward/back
    vy = js.axis(0, 0.0) * _VEL_SCALE_Y * -1        # axis 0 → lateral (invert)
    yaw = js.axis(3, 0.0) * _VEL_SCALE_YAW * -1   # axis 3 → yaw rate (right stick L/R)
    cmd = torch.tensor([[vx, vy, yaw]], device=device)
    return cmd


# ── Main interactive loop ──────────────────────────────────────


def run_interactive(
    device: str,
    vel_fn: Callable,
    safefall_fn: Callable,
    host_fn: Callable,
    predictor,
    env_cfg_vel,
    env_cfg_safefall,
    env_cfg_host,
    js_device: str = "/dev/input/js0",
) -> None:
    """Single‑viewer interactive demo with joystick control."""

    # ── Joystick ─────────────────────────────────────────
    js = JoystickReader(js_device)
    if not js.start():
        print(f"[WARN] Joystick {js_device} not found — falling back to keyboard.")
        print("       Use A/B keys with the passive viewer key callback.")
        _run_keyboard_fallback(
            device, vel_fn, safefall_fn, host_fn, predictor,
            env_cfg_vel, env_cfg_safefall, env_cfg_host,
        )
        return

    print(f"[INFO] Joystick connected: {js_device}")
    print("       Left stick → velocity  |  A → push  |  B → stand‑up  |  Esc → quit")
    print()

    # ── Envs ─────────────────────────────────────────────
    cfg_walk = env_cfg_vel(play=True)
    cfg_walk.scene.num_envs = 1
    walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
    asset = walk_env.scene["robot"]

    _cfg_fall = env_cfg_safefall(play=True)
    _cfg_fall.scene.num_envs = 1
    _cfg_stand = env_cfg_host(play=True)
    _cfg_stand.scene.num_envs = 1

    predictor._env = walk_env
    rng = np.random.default_rng()

    viewer = mujoco.viewer.launch_passive(
        walk_env.sim.mj_model, walk_env.sim.mj_data, key_callback=None,
    )

    walk_env.reset()
    predictor.reset()

    phase = "WALKING"
    step = 0
    phase_env = walk_env
    fell_env = None

    # Suppress ghost events from startup (all buttons + axis noise).
    for _btn in range(16):
        _ = js.button_oneshot(_btn)

    # ── Phase transition helpers ────────────────────────

    def _switch_to_falling():
        nonlocal phase, phase_env, fell_env, asset
        print("\n  ⚠  FALL DETECTED — SafeFall activated")
        state = capture_state(phase_env)
        phase_env.close()

        fenv = ManagerBasedRlEnv(_cfg_fall, device=device)
        fenv.reset()
        restore_state(fenv, state)
        predictor._env = fenv

        fell_env = fenv
        phase_env = fenv
        asset = fenv.scene["robot"]
        viewer._model = fenv.sim.mj_model
        viewer._data = fenv.sim.mj_data
        return "FALLING"

    def _switch_to_standing(from_env):
        nonlocal phase, phase_env, asset
        print("\n  🦿 Stand‑up activated")
        state = capture_state(from_env)
        from_env.close()

        senv = ManagerBasedRlEnv(_cfg_stand, device=device)
        senv.reset()
        restore_state(senv, state)

        phase_env = senv
        asset = senv.scene["robot"]
        viewer._model = senv.sim.mj_model
        viewer._data = senv.sim.mj_data
        return "STANDING"

    def _switch_to_walking(from_env):
        nonlocal phase, phase_env, walk_env, asset, fell_env
        print("\n  ✅ STOOD UP — back to walking")
        state = capture_state(from_env)
        from_env.close()

        walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
        walk_env.reset()
        restore_state(walk_env, state)
        predictor.reset()
        predictor._env = walk_env

        phase_env = walk_env
        asset = walk_env.scene["robot"]
        fell_env = None
        viewer._model = walk_env.sim.mj_model
        viewer._data = walk_env.sim.mj_data
        return "WALKING"

    # ── Labels dictionary ───────────────────────────────
    phase_labels = {"WALKING": "WALK ", "FALLING": "FALL ", "STANDING": "STAND"}

    print(f"  {step:5d} [WALK] ready — use left stick to walk")

    # ── Main loop (Esc disabled — closes only on window close) ──
    while True:
        step += 1
        bh = asset.data.root_link_pos_w[0, 2].item()

        # ── JOYSTICK INPUT ──────────────────────────
        if js.button_oneshot(0):  # A — push
            if phase == "WALKING":
                fx, fy = _random_push(rng)
                _apply_push(asset, fx, fy, device)
                print(f"  💨 PUSH  fx={fx:+.0f}  fy={fy:+.0f} N")
            else:
                print(f"  [push ignored — not walking]")

        if js.button_oneshot(1):  # B — stand‑up
            if phase in ("WALKING", "FALLING"):
                phase = _switch_to_standing(phase_env)
            else:
                print("  [already standing]")

        if js.button_oneshot(2):  # X — loco
            if phase in ("FALLING", "STANDING"):
                phase = _switch_to_walking(phase_env)
            else:
                print("  [already walking]")

        if js.button_oneshot(6):  # Select/Back — reset env
            print("  🔄 Reset")
            phase_env.reset()
            predictor.reset()

        # ── PREDICTOR ────────────────────────────────
        if phase == "WALKING":
            is_falling, prob = predictor.update(None)
            if is_falling:
                phase = _switch_to_falling()

        # ── VELOCITY COMMAND ─────────────────────────
        cmd = _js_to_cmd(js, device)

        # ── POLICY STEP ──────────────────────────────
        if phase == "WALKING":
            obs = {k: v.to(device) for k, v in
                   phase_env.get_observations().items()
                   if k in ("actor", "critic")}
            obs["actor"][:, -3:] = cmd
            phase_env.step(vel_fn(obs))

        elif phase == "FALLING":
            obs = {k: v.to(device) for k, v in
                   phase_env.get_observations().items()
                   if k in ("actor", "critic")}
            phase_env.step(safefall_fn(obs))

        elif phase == "STANDING":
            obs = {k: v.to(device) for k, v in
                   phase_env.get_observations().items()
                   if k in ("actor", "critic")}
            phase_env.step(host_fn(obs))

            # Auto‑return to walking once stood up.
            if bh > 0.65 and hasattr(phase_env.scene["robot"], "data"):
                phase = _switch_to_walking(phase_env)

        try:
            _sync(phase_env, viewer)
        except Exception:
            break  # viewer window closed

        # ── Periodic status ──────────────────────────
        if step % 200 == 0:
            label = phase_labels.get(phase, "????")
            p_str = f"  prob={predictor.probability:.2f}" if hasattr(predictor, 'probability') else ""
            vmag = (cmd[0, 0]**2 + cmd[0, 1]**2)**0.5
            print(f"  {step:5d} [{label}] base_h={bh:.3f}  |v|={vmag:.2f}{p_str}")

    # ── Cleanup ────────────────────────────────────────────
    js.stop()
    if fell_env is not None:
        fell_env.close()
    if phase == "STANDING":
        phase_env.close()
    else:
        walk_env.close() if phase == "WALKING" else phase_env.close()
    viewer.close()


# ── Keyboard fallback (no joystick) ────────────────────────────


def _run_keyboard_fallback(
    device, vel_fn, safefall_fn, host_fn, predictor,
    env_cfg_vel, env_cfg_safefall, env_cfg_host,
) -> None:
    """Fallback: keyboard A/B keys via MuJoCo viewer callback."""

    class _Keys:
        def __init__(self):
            self.push = False
            self.standup = False
            self.loco = False

        def callback(self, keycode: int):
            key = chr(keycode) if 32 <= keycode < 127 else ""
            if key in ("a", "A"):
                self.push = True
            elif key in ("b", "B"):
                self.standup = True
            elif key in ("x", "X"):
                self.loco = True

    keys = _Keys()
    cfg_walk = env_cfg_vel(play=True)
    cfg_walk.scene.num_envs = 1
    walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
    asset = walk_env.scene["robot"]

    _cfg_fall = env_cfg_safefall(play=True)
    _cfg_fall.scene.num_envs = 1
    _cfg_stand = env_cfg_host(play=True)
    _cfg_stand.scene.num_envs = 1

    predictor._env = walk_env
    rng = np.random.default_rng()

    viewer = mujoco.viewer.launch_passive(
        walk_env.sim.mj_model, walk_env.sim.mj_data,
        key_callback=keys.callback,
    )

    walk_env.reset()
    predictor.reset()

    phase = "WALKING"
    step = 0
    phase_env = walk_env
    fell_env = None

    cmd = torch.zeros(1, 3, device=device)  # zero velocity by default

    print("  Keyboard mode: [A] push  [B] stand‑up  [X] loco")
    print()

    while True:
        step += 1
        bh = asset.data.root_link_pos_w[0, 2].item()

        # Push
        if keys.push:
            keys.push = False
            if phase == "WALKING":
                fx, fy = _random_push(rng)
                _apply_push(asset, fx, fy, device)
                print(f"  💨 PUSH  fx={fx:+.0f}  fy={fy:+.0f} N")

        # Stand‑up
        if keys.standup:
            keys.standup = False
            if phase in ("FALLING", "WALKING"):
                state = capture_state(phase_env)
                phase_env.close()
                senv = ManagerBasedRlEnv(_cfg_stand, device=device)
                senv.reset()
                restore_state(senv, state)
                phase_env = senv
                asset = senv.scene["robot"]
                viewer._model = senv.sim.mj_model
                viewer._data = senv.sim.mj_data
                phase = "STANDING"
                print("  🦿 Stand‑up activated")
            elif phase == "STANDING":
                if bh > 0.65:
                    state = capture_state(phase_env)
                    phase_env.close()
                    walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
                    walk_env.reset()
                    restore_state(walk_env, state)
                    predictor.reset()
                    predictor._env = walk_env
                    phase_env = walk_env
                    asset = walk_env.scene["robot"]
                    viewer._model = walk_env.sim.mj_model
                    viewer._data = walk_env.sim.mj_data
                    phase = "WALKING"
                    fell_env = None
                    print("  ✅ Back to walking")

        # Loco (X key)
        if keys.loco:
            keys.loco = False
            if phase in ("FALLING", "STANDING"):
                state = capture_state(phase_env)
                phase_env.close()
                walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
                walk_env.reset()
                restore_state(walk_env, state)
                predictor.reset()
                predictor._env = walk_env
                phase_env = walk_env
                asset = walk_env.scene["robot"]
                fell_env = None
                viewer._model = walk_env.sim.mj_model
                viewer._data = walk_env.sim.mj_data
                phase = "WALKING"
                print("  🏃 Loco policy activated")

        # Predictor
        if phase == "WALKING":
            is_falling, prob = predictor.update(None)
            if is_falling:
                print("\n  ⚠  FALL DETECTED — SafeFall activated")
                state = capture_state(phase_env)
                phase_env.close()
                fenv = ManagerBasedRlEnv(_cfg_fall, device=device)
                fenv.reset()
                restore_state(fenv, state)
                predictor._env = fenv
                fell_env = fenv
                phase_env = fenv
                asset = fenv.scene["robot"]
                viewer._model = fenv.sim.mj_model
                viewer._data = fenv.sim.mj_data
                phase = "FALLING"

        # Policy step
        obs = {k: v.to(device) for k, v in phase_env.get_observations().items()
               if k in ("actor", "critic")}
        if phase == "WALKING":
            obs["actor"][:, -3:] = cmd
            phase_env.step(vel_fn(obs))
        elif phase == "FALLING":
            phase_env.step(safefall_fn(obs))
        elif phase == "STANDING":
            phase_env.step(host_fn(obs))
            if bh > 0.65:
                state = capture_state(phase_env)
                phase_env.close()
                walk_env = ManagerBasedRlEnv(cfg_walk, device=device)
                walk_env.reset()
                restore_state(walk_env, state)
                predictor.reset()
                predictor._env = walk_env
                phase_env = walk_env
                asset = walk_env.scene["robot"]
                fell_env = None
                viewer._model = walk_env.sim.mj_model
                viewer._data = walk_env.sim.mj_data
                phase = "WALKING"
                print("  ✅ Stood up — back to walking")

        try:
            _sync(phase_env, viewer)
        except Exception:
            break

        if step % 200 == 0:
            p_str = f" prob={predictor.probability:.2f}" if hasattr(predictor, 'probability') else ""
            print(f"  {step:5d} [{phase[:4]:4s}] base_h={bh:.3f}{p_str}")

    if fell_env is not None:
        fell_env.close()
    if phase == "STANDING":
        phase_env.close()
    else:
        walk_env.close() if phase == "WALKING" else phase_env.close()
    viewer.close()
