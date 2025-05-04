import argparse
from src.dataloading import get_train_val_dataset
from src.models.lstms import SimpleLSTM
from src.training_utils import build_optimizer, create_build_dataloaders, run_sweep

from src.transforms import fft_filtering

def main(config_file_path, wandb_project_name, count, seed, checkpoint_freq):
    dataset_tr, dataset_val = get_train_val_dataset(
        tr_ratio_min=0.8,
        tr_ratio_max=0.85,
        seed=seed,
        signal_transform=fft_filtering,
        label_transform=None,
        prefetch=True,
        resample_label=False
    )

    build_dataloaders_fn = create_build_dataloaders(dataset_tr, dataset_val)

    sweep_id = run_sweep(
        project_name=wandb_project_name,
        config_path=config_file_path,
        seed=seed,
        sweep_run_count=count,
        checkpoint_freq=checkpoint_freq,
        model_class=SimpleLSTM,
        build_dataloaders_fn=build_dataloaders_fn,
        build_optimizer_fn=build_optimizer,
    )

    return sweep_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSTM sweep with WandB.")
    parser.add_argument("config_file_path", type=str, help="Path to the YAML config file.")
    parser.add_argument("--project_name", type=str, default="pytorch-lstm-sweep", help="Name of the WandB project.")
    parser.add_argument("--count", type=int, default=20, help="Number of runs for the sweep.")
    parser.add_argument("--seed", type=int, default=1, help="Seed to use for the experiments")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="The number of epochs between each checkpoint")
    args = parser.parse_args()

    main(args.config_file_path, args.project_name, args.count, args.seed, args.checkpoint_freq)