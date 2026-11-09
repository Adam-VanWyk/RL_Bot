"""
Single-agent Gymnasium wrapper around RLGym v2 + RocketSim (headless, no Rocket League needed).

The learning agent is the blue car in a 1v1. The orange car is controlled by a simple
scripted opponent (default: does nothing).
"""
import random
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlgym.api import RLGym, RewardFunction
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_GOAL_BACK, BLUE_TEAM
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

TICK_SKIP = 8  # physics runs at 120 Hz, so the agent acts 15 times per game second


# ---------------------------------------------------------------- reward shaping
class BallToGoalVelocityReward(RewardFunction[Any, GameState, float]):
    """Reward for ball velocity pointed at the opponent's goal (normalized to about [-1, 1])."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> Dict[Any, float]:
        out = {}
        for agent in agents:
            car = state.cars[agent]
            goal = np.array(ORANGE_GOAL_BACK if car.team_num == BLUE_TEAM else (0, -ORANGE_GOAL_BACK[1], ORANGE_GOAL_BACK[2]))
            direction = goal - state.ball.position
            direction /= np.linalg.norm(direction) + 1e-6
            out[agent] = float(np.dot(state.ball.linear_velocity, direction) / 6000.0)
        return out


class SpeedTowardBallReward(RewardFunction[Any, GameState, float]):
    """Small reward for driving toward the ball, so the agent discovers touching it."""

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info) -> Dict[Any, float]:
        out = {}
        for agent in agents:
            car = state.cars[agent].physics
            to_ball = state.ball.position - car.position
            to_ball /= np.linalg.norm(to_ball) + 1e-6
            out[agent] = float(max(0.0, np.dot(car.linear_velocity, to_ball)) / 2300.0)
        return out


# ---------------------------------------------------------------- environment
class RocketLeagueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, timeout_seconds: float = 20.0, no_touch_seconds: float = 8.0, opponent: str = "noop"):
        super().__init__()
        self._opponent_mode = opponent

        self._lookup = LookupTableAction.make_lookup_table()
        self._noop_index = int(np.where((self._lookup == 0).all(axis=1))[0][0])

        self._rlgym = RLGym(
            state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator()),
            obs_builder=DefaultObs(zero_padding=None),
            action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
            reward_fn=CombinedReward(
                (GoalReward(), 10.0),
                (TouchReward(), 0.5),
                (BallToGoalVelocityReward(), 1.0),
                (SpeedTowardBallReward(), 0.1),
            ),
            termination_cond=GoalCondition(),
            truncation_cond=AnyCondition(TimeoutCondition(timeout_seconds), NoTouchTimeoutCondition(no_touch_seconds)),
            transition_engine=RocketSimEngine(),
        )

        obs = self._rlgym.reset()
        self._agents: List[Any] = list(obs.keys())
        # The learning agent is the blue car, the other one is the opponent.
        self._me = next(a for a in self._agents if self._rlgym.state.cars[a].team_num == BLUE_TEAM)
        self._opp = next(a for a in self._agents if a != self._me)

        obs_dim = int(obs[self._me].shape[0])
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(len(self._lookup))  # 90

    def _opponent_action(self) -> int:
        if self._opponent_mode == "random":
            return int(self.np_random.integers(len(self._lookup)))
        return self._noop_index

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)  # KickoffMutator picks spawns with Python's global `random`
        obs = self._rlgym.reset()
        return obs[self._me].astype(np.float32), {}

    def step(self, action):
        actions = {
            self._me: np.array([int(action)]),
            self._opp: np.array([self._opponent_action()]),
        }
        obs, rewards, terminated, truncated = self._rlgym.step(actions)
        state = self._rlgym.state
        info = {
            "touched": bool(state.cars[self._me].ball_touches > 0),
            "goal_for": bool(state.goal_scored and state.scoring_team == BLUE_TEAM),
            "goal_against": bool(state.goal_scored and state.scoring_team != BLUE_TEAM),
        }
        return (
            obs[self._me].astype(np.float32),
            float(rewards[self._me]),
            bool(terminated[self._me]),
            bool(truncated[self._me]),
            info,
        )

    def close(self):
        self._rlgym.close()
