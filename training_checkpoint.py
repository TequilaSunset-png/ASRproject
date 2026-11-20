"""Epoch-boundary checkpoint support shared by the training entry points."""

import argparse
from pathlib import Path

import torch


def parse_training_args(default_project, run_name):
    parser = argparse.ArgumentParser(description="Train Georgian HuBERT CTC")
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--epochs", type=int, default=40,
                        help="Total target epoch count, including completed epochs")
    parser.add_argument("--resume", type=Path,
                        help="Training checkpoint (.pt) to resume")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Directory for latest.pt (default: outputs/<run>_checkpoints)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save every N epochs; always save the final epoch")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for the final Hugging Face model and processor")
    args = parser.parse_args()
    if args.epochs < 1 or args.save_every < 1:
        parser.error("--epochs and --save-every must be positive")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.project / "outputs" / f"{run_name}_checkpoints"
    if args.output_dir is None:
        args.output_dir = args.project / "outputs" / f"{run_name}_final"
    return args


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    # Keep the last successful checkpoint intact if writing is interrupted.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)
    print(f"Checkpoint saved: {path} (epoch={epoch}, step={global_step})", flush=True)


def restore_checkpoint(path, model, optimizer, scheduler, metadata):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint format")
    if state["metadata"] != metadata:
        raise ValueError("Checkpoint training settings/tokenizer do not match this run")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    print(f"Resumed {path}: completed epoch={state['epoch']}, "
          f"step={state['global_step']}", flush=True)
    return state["epoch"] + 1, state["global_step"]
