# Copyright (c) 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import time
from types import SimpleNamespace

import numpy as np
from kspdg.lbg1.lg1_envs import LBG1_LG1_ParentEnv


def test_short_burn_wait_clamps_sleep_below_poll_quantum():
    """A 0.01 s burn must clamp its wait sleep instead of overshooting by 50 ms."""
    env = LBG1_LG1_ParentEnv(loadfile="lbg1_i1_init", reorient_time=0.0)
    env.bot_thread_status = env.BOT_HEALTHY_FLAG
    env.is_episode_done = False
    env.get_observation = lambda: np.zeros(12, dtype=float)
    env.get_info = lambda obs, done: {env.PARAMS.INFO.K_WEIGHTED_SCORE: 0.0}
    env.get_reward = lambda info, done: 0.0

    ves_agent = SimpleNamespace(control=SimpleNamespace(forward=0.0, right=0.0, up=0.0))

    t0 = time.perf_counter()
    env.vessel_step([1.0, 0.0, 0.0, 0.01], ves_agent)
    elapsed = time.perf_counter() - t0

    assert 0.008 <= elapsed < 0.035
    assert ves_agent.control.forward == 0.0
    assert ves_agent.control.right == 0.0
    assert ves_agent.control.up == 0.0


def test_guard_bot_loops_check_breakout_before_sleeping():
    """Triggered guard maneuver transitions must break immediately without a 50 ms lag."""
    env = LBG1_LG1_ParentEnv(loadfile="lbg1_i1_init", reorient_time=0.0)
    env.stop_bot_thread = False
    env.vesGuard = SimpleNamespace(
        orbital_reference_frame=object(),
        control=SimpleNamespace(forward=1.0),
        auto_pilot=SimpleNamespace(target_direction=None, error=0.0),
    )

    # 1. Zero-out breakout already met (spd < min_speed_thresh)
    env.vesBandit = SimpleNamespace(
        velocity=lambda _rf: (0.01, 0.0, 0.0),
        position=lambda _rf: (100.0, 0.0, 0.0),
    )
    t0 = time.perf_counter()
    env.zeroout_bandit_guard_velocity()
    assert time.perf_counter() - t0 < 0.025

    # 2. Direct burn breakout already met (spd > max_speed_thresh)
    env.vesBandit = SimpleNamespace(
        velocity=lambda _rf: (-15.0, 0.0, 0.0),
        position=lambda _rf: (100.0, 0.0, 0.0),
    )
    t0 = time.perf_counter()
    env.direct_burn_guard_toward_bandit()
    assert time.perf_counter() - t0 < 0.025

    # 3. Coast breakout already met (misalignment > ang_vel_pos_thresh)
    env.vesBandit = SimpleNamespace(
        velocity=lambda _rf: (0.0, 5.0, 0.0),
        position=lambda _rf: (100.0, 0.0, 0.0),
    )
    t0 = time.perf_counter()
    env.coast_until_pos_vel_misalign()
    assert time.perf_counter() - t0 < 0.025
