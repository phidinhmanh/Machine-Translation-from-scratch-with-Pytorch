import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import gc
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
        "--vocab_size", type=int, default=8000, help="Vocab size (khớp với SPM)"
    )

    # Model params
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
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

    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        src = batch["src_ids"].to(device)
        tgt = batch["tgt_ids"].to(device)

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        if step % 4 == 0:
            optimizer.zero_grad()

        with torch.amp.autocast_mode.autocast(
            enabled=(device == "cuda"), device_type=device
        ):
            output = model(src, decoder_input)

            # print(f"Is NaN: {torch.isnan(output).any()}")
            # print(f"min and max: {output.min().item()}, {output.max().item()}")

            # Reshape để tính Loss
            # Output: [Batch * Seq_Len, Vocab]
            # Labels: [Batch * Seq_Len]
            loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

        # 4. Backward Pass (Backward + Optimize)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # 5. Logging
        total_loss += loss.item()

        # check gradient
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Gradient is None for {name}")
                break
            if param.grad.isnan().any():
                print(f"Gradient is NaN for {name}")
                break
            if param.grad.sum() == 0:
                print(f"Gradient is zero for {name}")
                break
        # check memory available
        # print(f"Memory available: {torch.cuda.memory_allocated() / 1e9}")

        # check loss is nan
        if torch.isnan(loss):
            print("Loss is NaN")
            break
        # if step == 200:
        #     return total_loss / 200

    return total_loss / len(loader)


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

    # Đếm tham số chơi cho vui
    print(
        f"Test Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 3. Setup Training
    # AdamW là chân ái cho Transformer (Weight Decay giúp chống Overfit)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Loss function: Ignore index 0 (PAD) để không tính điểm cho padding
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_idx, label_smoothing=0.1)

    # Scaler cho Mixed Precision
    scaler = torch.amp.grad_scaler.GradScaler(enabled=(args.device == "cuda"))

    # Scheduler (Giảm LR khi Loss không giảm nữa)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    print("🔥 Start Training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, args.device, epoch
        )
        gc.collect()
        torch.cuda.empty_cache()
        # Validate
        val_loss = evaluate(model, val_loader, criterion, args.device)
        # Update LR
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "best_transformer.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model to {save_path}")

    print("🎉 Training Completed!")


if __name__ == "__main__":
    main()
