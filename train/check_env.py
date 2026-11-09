"""Sanity check + statistics for the proposal: run from the repo root with `python -m train.check_env`."""
import time
import numpy as np
from gymnasium.utils.env_checker import check_env
from train.rocketsim_env import RocketLeagueEnv, TICK_SKIP

env = RocketLeagueEnv()
print("observation space:", env.observation_space)
print("action space:     ", env.action_space)
check_env(env, skip_render_check=True)
print("gymnasium check_env passed")

lengths, rewards, t0, steps = [], [], time.perf_counter(), 0
for ep in range(50):
    env.reset(seed=ep)
    n, total = 0, 0.0
    while True:
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        n += 1; steps += 1; total += r
        if term or trunc:
            break
    lengths.append(n); rewards.append(total)
elapsed = time.perf_counter() - t0
print(f"{steps / elapsed:.0f} agent steps/s  ({steps * TICK_SKIP / elapsed:.0f} physics ticks/s)")
print(f"{elapsed / 50:.3f} s per episode")
print(f"episode length: mean {np.mean(lengths):.0f}, min {min(lengths)}, max {max(lengths)} agent steps")
print(f"mean episode return (random policy): {np.mean(rewards):.2f}")
