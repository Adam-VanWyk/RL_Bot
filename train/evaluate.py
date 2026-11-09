"""
Evaluate a trained model against the scripted opponent.
    python -m train.evaluate --model runs/e1_baseline/final_model.zip --episodes 50
Prints goals scored, goals conceded, and the fraction of episodes with a ball touch.
"""
import argparse
import numpy as np
from stable_baselines3 import PPO

from train.rocketsim_env import RocketLeagueEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--opponent", default="noop", choices=["noop", "random"])
    args = p.parse_args()

    env = RocketLeagueEnv(opponent=args.opponent)
    model = PPO.load(args.model, device="cpu")
    scored = conceded = touched = 0
    lengths = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        n, touch = 0, False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(int(action))
            n += 1
            touch = touch or info.get("touched", False)
            if term or trunc:
                scored += int(info.get("goal_for", False))
                conceded += int(info.get("goal_against", False))
                break
        touched += int(touch)
        lengths.append(n)
    print(f"{args.episodes} episodes vs {args.opponent} opponent")
    print(f"goals scored: {scored}, goals conceded: {conceded}")
    print(f"episodes with a ball touch: {touched}/{args.episodes}")
    print(f"mean episode length: {np.mean(lengths):.0f} agent steps")


if __name__ == "__main__":
    main()
