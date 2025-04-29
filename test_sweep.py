import argparse
from sklearn import set_config
import wandb
from src.dataloading import get_dummy_pickle
from src.models.lstms import SimpleLSTM
from src.training_utils import build_optimizer, create_train_fn
import yaml


def build_lstm(config):
    return SimpleLSTM(
        input_dim=19,
        hidden_dim=config.hidden_dim,
        num_layers=1,
        dropout=0.2
    )

def build_dummy_dataset(wandb_config):
    tr = get_dummy_pickle()
    val = get_dummy_pickle()
    return tr, val

def main(config_file_name, project_name, count):
    with open(f"configs/{config_file_name}", "r") as file:
        config = yaml.safe_load(file)

    train_fn = create_train_fn(
        build_dataset_fn=build_dummy_dataset, 
        build_network_fn=build_lstm, 
        build_optimizer_fn=build_optimizer
    )

    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id, train_fn, count=count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM sweep with WandB.")
    parser.add_argument("config_file_name", type=str, help="Path to the YAML config file.")
    parser.add_argument("--project_name", type=str, default="pytorch-lstm-sweep", help="Name of the WandB project.")
    parser.add_argument("--count", type=int, default=20, help="Number of runs for the sweep.")
    args = parser.parse_args()

    main(args.config_file_name, args.project_name, args.count)