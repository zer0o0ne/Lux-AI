from self_playing import play
from dataset import DatasetBuilder
from torch.utils.data import DataLoader
from trainer import Trainer
from estimator import Estimator

agent_cfg_1 = {
    "estimator_n_iter": 16,
    "MMCTS_n_iter": 64,
    "game_n": 1,
    "T": 1,
    "need_save": True,
    "weights_path": 'solution/weights/1_weights_1'
}

agent_cfg_2 = {
    "estimator_n_iter": 16,
    "MMCTS_n_iter": 64,
    "game_n": 1,
    "T": 1,
    "need_save": True,
    "weights_path": 'solution/weights/1_weights_2'
}

n_cycles, n_games, batch_size = 500, 20, 512

for cycle in range(10, n_cycles):
    agent_cfg_1, agent_cfg_2 = play(n_games, agent_cfg_1=agent_cfg_1, agent_cfg_2=agent_cfg_2)
    model_1 = Estimator(agent_cfg_1["estimator_n_iter"])
    model_2 = Estimator(agent_cfg_2["estimator_n_iter"])
    trainer = Trainer(agent_cfg_1["weights_path"], model_1, "player_0", n_games = n_games * 3)
    trainer.train(batch_size = batch_size, cycle = cycle)
    del trainer
    trainer = Trainer(agent_cfg_2["weights_path"], model_1, "player_1", n_games = n_games * 3)
    trainer.train(batch_size = batch_size, cycle = cycle)
    del trainer
    agent_cfg_1["weights_path"] = f'solution/weights/{cycle + 1}_weights_1'
    agent_cfg_2["weights_path"] = f'solution/weights/{cycle + 1}_weights_2'
