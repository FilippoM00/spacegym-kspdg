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
