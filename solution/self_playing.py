import sys
from time import time
sys.path.append("kits/python/")
import json
from tqdm.auto import tqdm

from kits.python.luxai_s2.env import LuxAI_S2

sys.path.append("kits/python/solution")

from agent import MMCTS_Agent, AlgorithmicAgent

def play(n_games, n_rounds = 1000, agent_cfg_1 = None, agent_cfg_2 = None):
    if agent_cfg_1 is None:
        agent_cfg_1 = {
            "estimator_n_iter": 16,
            "MMCTS_n_iter": 64,
            "game_n": 1,
            "T": 1,
            "need_save": True,
            "weights_path": 'solution/weights_1'
        }

    if agent_cfg_2 is None:
        agent_cfg_2 = {
            "estimator_n_iter": 16,
            "MMCTS_n_iter": 64,
            "game_n": 1,
            "T": 1,
            "need_save": True,
            "weights_path": 'solution/weights_2'
        }

    for n in tqdm(range(n_games), desc = "games in one cycle"):
        env = LuxAI_S2()
        obs = env.reset()["player_0"]
        # env.env_cfg.verbose = 0
        agent_1 = MMCTS_Agent("player_0", env.env_cfg, agent_cfg_1) if agent_cfg_1["agent_type"] == "MMCTS" else AlgorithmicAgent("player_0", env.env_cfg, agent_cfg_1)
        agent_2 = MMCTS_Agent("player_1", env.env_cfg, agent_cfg_2) if agent_cfg_2["agent_type"] == "MMCTS" else AlgorithmicAgent("player_1", env.env_cfg, agent_cfg_2)
        steps_before = -obs["real_env_steps"]

        action = {"player_0": agent_1.early_setup(0, obs), "player_1": agent_2.early_setup(0, obs)}
        obs = env.step(action)[0]["player_0"]

        while obs["real_env_steps"] < 0:
            action = {"player_0": agent_1.early_setup(steps_before + obs["real_env_steps"], obs), "player_1": agent_2.early_setup(steps_before + obs["real_env_steps"], obs)}
            obs = env.step(action)[0]["player_0"]

        for step in range(steps_before, steps_before + n_rounds): 
            # time_1 = time()
            act_1 = agent_1.act(step, obs)
            # time_1 = time() - time_1
            # time_2 = time()
            act_2 = agent_2.act(step, obs)
            # time_2 = time() - time_2
            action = {"player_0": act_1, "player_1": act_2}
            obs, rewards, dones, _ = env.step(action)
            obs = obs["player_0"]
            if dones["player_0"] or dones["player_1"]:
                break
            # print(f'success, agent_1 time: {time_1}, agent_2 time: {time_2}')

        path_0 = agent_1.MCTS.path + "/player_0_reward.json"
        path_1 = agent_2.MCTS.path + "/player_1_reward.json"
        path_2 = agent_2.MCTS.path + "/player_0_reward.json"
        path_3 = agent_1.MCTS.path + "/player_1_reward.json"

        with open(path_0, 'w') as fp:
            if rewards["player_0"] > rewards["player_1"]:
                rew = 1
            elif rewards["player_0"] < rewards["player_1"]:
                rew = -1
            else:
                rew = -0.5
            json.dump(rew, fp)

        with open(path_1, 'w') as fp:
            if rewards["player_0"] < rewards["player_1"]:
                rew = 1
            elif rewards["player_0"] > rewards["player_1"]:
                rew = -1
            else:
                rew = -0.5
            json.dump(rew, fp)

        with open(path_2, 'w') as fp:
            if rewards["player_0"] > rewards["player_1"]:
                rew = 1
            elif rewards["player_0"] < rewards["player_1"]:
                rew = -1
            else:
                rew = -0.5
            json.dump(rew, fp)

        with open(path_3, 'w') as fp:
            if rewards["player_0"] < rewards["player_1"]:
                rew = 1
            elif rewards["player_0"] > rewards["player_1"]:
                rew = -1
            else:
                rew = -0.5
            json.dump(rew, fp)

        agent_cfg_1["game_n"] += 1
        agent_cfg_2["game_n"] += 1

    return agent_cfg_1, agent_cfg_2