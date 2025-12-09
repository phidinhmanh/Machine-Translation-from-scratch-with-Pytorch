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
    parser.add_argument("--heads", type=int, default=8)
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
        "step": step,  # Lưu step kế tiếp cần chạy
        "best_val_loss": best_val_loss,
        "elapsed_time": elapsed_time,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    print(f"💾 Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """Load training checkpoint to resume training."""
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    # weights_only=False để load full object (cần thiết cho optimizer/scheduler)
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
        # --- FIX: LOGIC SKIP CHUẨN ---
        # start_step được load từ checkpoint là (step cũ + 1)
        # Nếu step hiện tại < start_step, nghĩa là đã train rồi -> Skip
        if step < start_step:
            continue

        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        # Safety Check
        if src.shape[1] > 150 or tgt.shape[1] > 150:
            del src, tgt
            continue

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        with torch.amp.autocast_mode.autocast(
            enabled=(device == "cuda"), device_type=device
        ):
            output = model(src, decoder_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        loss_val = loss.item() * ACCUMULATION_STEPS
        total_loss += loss_val
        steps_counted += 1

        # Accumulation Update
        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)

            # Check Grad Norm
            grad_norm = 0.0
            first_layer_grad = list(model.parameters())[0].grad
            if first_layer_grad is not None:
                grad_norm = first_layer_grad.norm().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if (step + 1) % 1000 == 0:
                logger.info(
                    f"Step {step} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}"
                )
                print(
                    f"Step {step}/{len(loader)} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}"
                )

            # Checkpoint Callback
            if checkpoint_callback is not None:
                # --- FIX: LƯU STEP + 1 ---
                # Để lần sau resume sẽ bắt đầu từ step tiếp theo
                checkpoint_callback(epoch, step + 1)

        # Check NaN
        if torch.isnan(loss):
            print("❌ Loss is NaN")
            break

        # Cleanup
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

            del src, tgt, decoder_input, labels, output, loss

    return total_loss / len(loader)


def main():
    args = get_args()
    print(f"🚀 Config: {vars(args)}")
    os.makedirs(args.save_dir, exist_ok=True)

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

    print("🏗️ Initializing Model...")
    model = Transformer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        heads=args.heads,
        blocks=args.layers,
        dropout=args.dropout,
        ff_expansion=args.ff_expansion,
        device=args.device,
    ).to(args.device)

    # Init Weights (Optional but recommended for deep models)
    # model.apply(model._init_weights)

    print(
        f"Test Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Setup Training
    # lr=1.0 cho Noam Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def noam_lr_lambda(step):
        step_num = max(1, step)
        warmup_steps = 4000  # Tăng lên 8000 nếu model to hơn
        d_model = args.embed_dim
        return (d_model**-0.5) * min(step_num**-0.5, step_num * (warmup_steps**-1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx, label_smoothing=0.1)
    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.device == "cuda"))

    # State Variables
    best_val_loss = float("inf")
    start_epoch = 1
    start_step = 0
    previous_elapsed_time = 0

    # Resume Logic
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        start_epoch, start_step, best_val_loss, previous_elapsed_time = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, scaler, args.device
        )

        # Nếu step = 0, nghĩa là đã hoàn thành epoch trước -> Tăng epoch lên 1
        if start_step == 0:
            start_epoch += 1

        # Đảm bảo model ở chế độ train sau khi load
        model.train()

    # Timer Setup
    checkpoint_interval_seconds = args.checkpoint_interval_hours * 3600
    training_start_time = time.time()
    last_checkpoint_time = training_start_time

    def get_total_elapsed_time():
        return previous_elapsed_time + (time.time() - training_start_time)

    def checkpoint_callback(epoch, step):
        nonlocal last_checkpoint_time
        current_time = time.time()

        # Check time interval
        if (current_time - last_checkpoint_time) >= checkpoint_interval_seconds:
            total_elapsed = get_total_elapsed_time()

            # Save Time-based Checkpoint
            chk_path = os.path.join(
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
                chk_path,
            )

            # Save Latest (Overwrite)
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
            print(f"⏰ Checkpoint saved. Total time: {total_elapsed / 3600:.2f}h")

    print("🔥 Start Training...")

    for epoch in range(start_epoch, args.epochs + 1):
        # Xác định start_step cho epoch này
        # Nếu epoch hiện tại == start_epoch (lúc resume), thì dùng start_step đã load
        # Các epoch sau thì start_step luôn là 0
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

        # Dọn rác
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate
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

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, "best_transformer.pth")
            )
            print("✅ Saved Best Model")

        # Save End of Epoch Checkpoint (Step = 0 -> Đánh dấu hoàn thành epoch)
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
