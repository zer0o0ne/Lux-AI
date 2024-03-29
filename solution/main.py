import json
from typing import Dict
import sys
from argparse import Namespace
sys.path.append("kits/python/")

from agent import MMCTS_Agent
from kits.python.luxai_s2.config import EnvConfig
from kits.python.lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
agent_cfg = {
            "estimator_n_iter": 18,
            "MMCTS_n_iter": 36,
            "game_n": 1,
            "T": 1,
            "need_save": False,
            "weights_path": 'weights/7_weights_1'
        }

def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step
    
    
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = MMCTS_Agent(player, env_cfg, agent_cfg)
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if obs["real_env_steps"] < 0:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)

if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    configurations = None
    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)
        
        observation = Namespace(**dict(step=obs["step"], obs=json.dumps(obs["obs"]), remainingOverageTime=obs["remainingOverageTime"], player=obs["player"], info=obs["info"]))
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # if obs["obs"]["real_env_steps"] >= -1:
        #     with open("log.txt", "a") as f:
        #         f.write(f'{str(len(actions))}, {obs["player"]},  \n')
        #         for name, item in actions.items():
        #             f.write(f'{name}: {item} \n')
        #         f.write("\n")
        #         f.write("\n")
        # send actions to engine
        print(json.dumps(actions))