import sys, os, json
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import *
from augmentations import *

AUGMENTATIONS = [
    identity,
    ninety_degrees, 
    hundred_eighteen_degrees, 
    two_hundred_seventeen_degrees,
    main_diagonal_flip,
    side_diagonal_flip,
    vertical_flip,
    horizontal_flip
]

class DatasetBuilder(Dataset):
    def __init__(self, player, n_games = 200, device = "cpu"):
        super().__init__()
        games = os.listdir("solution/train")
        idx = np.argsort([int(name.split("_")[0]) for name in games])
        games = np.array(games)[idx]
        n_games = min(n_games, len(games))
        games = games[-n_games:]

        self.prepared_states, self.vs, self.policies = [], [], []
        for game in games:
            with open(f'solution/train/{game}/player_0_reward.json', 'r') as f:
                player_0_reward = torch.tensor(json.load(f), dtype = torch.float).to(device)
            with open(f'solution/train/{game}/player_1_reward.json', 'r') as f:
                player_1_reward = torch.tensor(json.load(f), dtype = torch.float).to(device)
            for file in os.listdir("solution/train/" + game):
                if "reward" in file:
                    continue
                with open(f'solution/train/{game}/{file}', 'r') as f:
                    data = json.load(f) 
                data["state"] = to_tensor(data["state"], device, type = "state")
                data["distribution"] = to_tensor(data["distribution"], device, type = "distribution")

                for augm in AUGMENTATIONS:
                    state, distribution = augm(data["state"], data["distribution"])
                    self.prepared_states.append(state)
                    self.vs.append({"player_0": player_0_reward, "player_1": player_1_reward})
                    self.policies.append(distribution)
        idx = np.random.permutation(len(self.vs))
        self.prepared_states, self.vs, self.policies = np.array(self.prepared_states), np.array(self.vs), np.array(self.policies)
        self.prepared_states, self.vs, self.policies = self.prepared_states[idx], self.vs[idx], self.policies[idx]

    def __len__(self):
        return len(self.vs)
    
    def __getitem__(self, idx):
        return self.prepared_states[idx], self.vs[idx], self.policies[idx]
    