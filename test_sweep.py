import argparse
from sklearn import set_config
import wandb
from src.dataloading import get_dummy_dataset, get_dummy_pickle, get_train_val_dataset
from src.models.lstms import SimpleLSTM
from src.training_utils import build_optimizer, create_build_dataloaders, create_train_fn, seed_everything
import yaml

from src.transforms import fft_filtering


def build_lstm(config):
    return SimpleLSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )


def main(config_file_name, project_name, count):
    with open(f"configs/{config_file_name}", "r") as file:
        config = yaml.safe_load(file)

    seed_everything(config.seed)

    dataset_tr, dataset_val = get_train_val_dataset(
        tr_ratio_min=0.8,
        tr_ratio_max=0.85,
        seed=config.seed,
        signal_transform=fft_filtering,
        label_transform=None,
        prefetch=True,
        resample_label=False
    )

    build_dataset = create_build_dataloaders(dataset_tr, dataset_val)

    train_fn = create_train_fn(
        build_dataset_fn=build_dataset, 
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