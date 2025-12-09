import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import gc
import time
from datasets import load_dataset
from model import Transformer
from preprocess import getdata_loader
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="train.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Train Transformer Vi-En")

    # Data params
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument(
        "--vocab_size", type=int, default=7000, help="Vocab size (khớp với SPM)"
    )

    # Model params
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--ff_expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoint_interval_hours",
        type=float,
        default=10.0,
        help="Save checkpoint every N hours (default: 10)",
    )

    return parser.parse_args()


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    step,
    best_val_loss,
    elapsed_time,
    save_path,
):
    """Save training checkpoint for resuming later."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "elapsed_time": elapsed_time,  # Total training time in seconds
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    print(f"💾 Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load training checkpoint to resume training."""
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"]
    start_step = checkpoint.get("step", 0)
    best_val_loss = checkpoint["best_val_loss"]
    elapsed_time = checkpoint.get("elapsed_time", 0)

    print(f"✅ Resumed from epoch {start_epoch}, step {start_step}")
    print(f"   Previous training time: {elapsed_time / 3600:.2f} hours")

    return start_epoch, start_step, best_val_loss, elapsed_time


def train_one_epoch(
    model,
    scheduler,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    start_step=0,
    checkpoint_callback=None,
):
    model.train()
    total_loss = 0
    steps_counted = 0

    optimizer.zero_grad()

    ACCUMULATION_STEPS = 4

    for step, batch in enumerate(loader):
        # Skip steps if resuming from a checkpoint
        if step < start_step:
            continue

        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        with torch.amp.autocast_mode.autocast(
            enabled=(device == "cuda"), device_type=device
        ):
            output = model(src, decoder_input)

            loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
            loss = loss / ACCUMULATION_STEPS
        # 4. Backward Pass (Backward + Optimize)
        scaler.scale(loss).backward()

        loss_val = loss.item() * ACCUMULATION_STEPS
        total_loss += loss_val
        steps_counted += 1

        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)

            # Calculate Grad Norm BEFORE clipping but AFTER unscaling
            grad_norm = 0.0
            first_layer_grad = list(model.parameters())[0].grad
            if first_layer_grad is not None:
                grad_norm = first_layer_grad.norm().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 1000 == 0:
                logger.info(
                    f"Step {step} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}"
                )

            # Call checkpoint callback for time-based saving
            if checkpoint_callback is not None:
                checkpoint_callback(epoch, step)

        # check memory available
        # print(f"Memory available: {torch.cuda.memory_allocated() / 1e9}")
        # torch.cuda.empty_cache() # Calling this too often slows down training

        # check loss is nan
        if torch.isnan(loss):
            print("Loss is NaN")
            break
        # if step == 200:
        #     return total_loss / 200
        del src, tgt, decoder_input, labels, output, loss

    return total_loss / max(steps_counted, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            src = batch["src_ids"].to(device)
            tgt = batch["tgt_ids"].to(device)

            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]

            output = model(src, decoder_input)

            loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    args = get_args()
    print(f"🚀 Config: {vars(args)}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    # Lưu ý: Em phải chắc chắn máy có internet để load opus100, hoặc trỏ vào local
    print("⏳ Loading Datasets...")
    datasets = load_dataset("opus100", "en-vi")
    train_loader = getdata_loader(
        datasets,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=True,
        type="train",
    )
    val_loader = getdata_loader(
        datasets,
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=False,
        type="validation",
    )

    # 2. Init Model
    print("🏗️ Initializing Model...")
    model = Transformer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        heads=args.heads,
        blocks=args.layers,
        dropout=args.dropout,
        ff_expansion=args.ff_expansion,  # Truyền 0 vào đây
        device=args.device,  # Cần thiết cho việc tạo mask
    ).to(args.device)

    print(
        f"Test Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 3. Setup Training
    # AdamW là chân ái cho Transformer (Weight Decay giúp chống Overfit)
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def noam_lr_lambda(step):
        step_num = max(1, step)
        warmup_steps = 4000
        d_model = args.embed_dim

        return (d_model**-0.5) * min(step_num**-0.5, step_num * (warmup_steps**-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lr_lambda)

    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx, label_smoothing=0.1)

    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.device == "cuda"))

    # Initialize training state
    best_val_loss = float("inf")
    start_epoch = 1
    start_step = 0
    previous_elapsed_time = 0  # Time from previous training sessions

    # Resume from checkpoint if specified
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        start_epoch, start_step, best_val_loss, previous_elapsed_time = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, scaler, args.device
        )
        # Continue from next epoch if step is 0, otherwise resume current epoch
        if start_step == 0:
            start_epoch += 1

    # Time-based checkpoint settings
    checkpoint_interval_seconds = (
        args.checkpoint_interval_hours * 3600
    )  # Convert hours to seconds
    training_start_time = time.time()
    last_checkpoint_time = training_start_time

    def get_total_elapsed_time():
        """Get total training time including previous sessions."""
        return previous_elapsed_time + (time.time() - training_start_time)

    def checkpoint_callback(epoch, step):
        """Callback to check if we should save a time-based checkpoint."""
        nonlocal last_checkpoint_time

        current_time = time.time()
        time_since_last_checkpoint = current_time - last_checkpoint_time

        if time_since_last_checkpoint >= checkpoint_interval_seconds:
            total_elapsed = get_total_elapsed_time()
            checkpoint_path = os.path.join(
                args.save_dir,
                f"checkpoint_epoch{epoch}_step{step}_time{total_elapsed / 3600:.1f}h.pth",
            )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                step,
                best_val_loss,
                total_elapsed,
                checkpoint_path,
            )
            # Also save as latest checkpoint for easy resume
            latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                step,
                best_val_loss,
                total_elapsed,
                latest_path,
            )
            last_checkpoint_time = current_time
            print(
                f"⏰ Time-based checkpoint saved. Total training time: {total_elapsed / 3600:.2f} hours"
            )

    print("🔥 Start Training...")
    print(f"⏰ Checkpoint will be saved every {args.checkpoint_interval_hours} hours")

    for epoch in range(start_epoch, args.epochs + 1):
        # Determine starting step for this epoch
        epoch_start_step = start_step if epoch == start_epoch else 0

        train_loss = train_one_epoch(
            model,
            scheduler,
            train_loader,
            optimizer,
            criterion,
            scaler,
            args.device,
            epoch,
            start_step=epoch_start_step,
            checkpoint_callback=checkpoint_callback,
        )
        gc.collect()
        torch.cuda.empty_cache()
        val_loss = evaluate(model, val_loader, criterion, args.device)
        current_lr = optimizer.param_groups[0]["lr"]
        total_time = get_total_elapsed_time()

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | Time: {total_time / 3600:.2f}h"
        )

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | Time: {total_time / 3600:.2f}h"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "best_transformer.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model to {save_path}")

        # Save checkpoint at end of each epoch
        epoch_checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            0,
            best_val_loss,
            get_total_elapsed_time(),
            epoch_checkpoint_path,
        )

    print("🎉 Training Completed!")
    print(f"⏰ Total training time: {get_total_elapsed_time() / 3600:.2f} hours")


if __name__ == "__main__":
    main()
