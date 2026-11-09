"""
PPO baseline on the RocketSim environment (stable-baselines3).

Run from the repo root, e.g.:
    python -m train.train --timesteps 200000 --n-envs 4 --run-name e1_baseline
Outputs go to runs/<run-name>/ (progress.csv, checkpoints, final_model.zip).
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from train.rocketsim_env import RocketLeagueEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4, help="parallel simulator processes (<= CPU cores)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95, help="bias/variance knob (experiment E3)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024, help="steps per env per PPO update")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--init-from", default=None, help="path to a model .zip to start from (e.g. a behavior-cloned policy)")
    args = p.parse_args()

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(RocketLeagueEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls)

    if args.init_from:
        model = PPO.load(args.init_from, env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
            gamma=args.gamma, gae_lambda=args.gae_lambda, seed=args.seed,
            policy_kwargs=dict(net_arch=[256, 256]), verbose=1, device="cpu",
        )
    model.set_logger(configure(out_dir, ["stdout", "csv"]))

    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1), save_path=out_dir, name_prefix="ckpt")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)
    model.save(os.path.join(out_dir, "final_model"))
    env.close()
    print("saved", os.path.join(out_dir, "final_model.zip"))


if __name__ == "__main__":
    main()
