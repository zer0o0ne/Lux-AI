from self_playing import play
from dataset import DatasetBuilder
from torch.utils.data import DataLoader
from trainer import Trainer
from estimator import Estimator

agent_cfg_1 = {
    "estimator_n_iter": 16,
    "MMCTS_n_iter": 64,
    "game_n": 1,
    "T": 0.3,
    "need_save": True,
    "weights_path": 'solution/weights/1_weights_1'
}

agent_cfg_2 = {
    "estimator_n_iter": 16,
    "MMCTS_n_iter": 64,
    "game_n": 1,
    "T": 0.3,
    "need_save": True,
    "weights_path": 'solution/weights/1_weights_2'
}

n_cycles, n_games, batch_size, start_cycle = 500, 5, 128, 1

for cycle in range(start_cycle, n_cycles + start_cycle):
    agent_cfg_1, agent_cfg_2 = play(n_games, agent_cfg_1=agent_cfg_1, agent_cfg_2=agent_cfg_2)
    model_1 = Estimator(agent_cfg_1["estimator_n_iter"])
    model_2 = Estimator(agent_cfg_2["estimator_n_iter"])
    trainer = Trainer(agent_cfg_1["weights_path"], model_1, "player_0", n_games = n_games * 6)
    trainer.train(batch_size = batch_size, cycle = cycle)
    del trainer
    trainer = Trainer(agent_cfg_2["weights_path"], model_1, "player_1", n_games = n_games * 6)
    trainer.train(batch_size = batch_size, cycle = cycle)
    del trainer
    agent_cfg_1["weights_path"] = f'solution/weights/{cycle}_weights_1'
    agent_cfg_2["weights_path"] = f'solution/weights/{cycle}_weights_2'
    
    
