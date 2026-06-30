"""Training / evaluation loop, shared by both variants.

`fit()` runs the full loop used in the notebook:
    - Adam (with weight decay) + MSE loss
    - ReduceLROnPlateau scheduler stepped on the train loss every epoch
    - validate every `eval_step` epochs, checkpoint on best val loss
    - early stop once best val loss drops below `stop_threshold`
"""
import time

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for geometry, spectrum in loader:
        geometry, spectrum = geometry.to(device), spectrum.to(device)
        optimizer.zero_grad()
        loss = criterion(model(geometry), spectrum)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for geometry, spectrum in loader:
        geometry, spectrum = geometry.to(device), spectrum.to(device)
        total += criterion(model(geometry), spectrum).item()
    return total / len(loader)


def fit(model, train_loader, val_loader, *, device, ckpt_path,
        epochs, lr, weight_decay, lr_decay_rate, lr_patience,
        eval_step, stop_threshold):
    """Run the full training loop. Returns (history, best_val_loss)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_decay_rate, patience=lr_patience)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    best_val = float("inf")
    history = {"epoch": [], "train": [], "val": []}
    start = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % eval_step == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            history["epoch"].append(epoch)
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {lr_now:.1e}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)

            if best_val < stop_threshold:
                print("Reached stop threshold, ending early.")
                break

        scheduler.step(train_loss)

    print(f"\nDone in {(time.time() - start) / 60:.1f} min. Best val loss: {best_val:.6f}")
    print(f"Saved best checkpoint to: {ckpt_path}")
    return history, best_val
