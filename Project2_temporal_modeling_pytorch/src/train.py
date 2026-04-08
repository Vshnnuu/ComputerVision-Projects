from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    BIDIRECTIONAL,
    CLASS_NAMES,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LAYERS,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from .data_prep import SequenceDataset
from .model import LSTMClassifier
from .utils import ensure_output_dir, plot_losses, print_report, save_confusion_matrix


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(X_train, X_test, y_train, y_test):
    set_seed(RANDOM_SEED)
    ensure_output_dir(OUTPUT_DIR)

    device = get_device()
    print(f"Using device: {device}")

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMClassifier(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_model_path = OUTPUT_DIR / "best_lstm_classifier.pt"

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                running_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    plot_losses(train_losses, val_losses, OUTPUT_DIR / "loss_curve.png")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    final_preds = []
    final_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)

            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(y_batch.cpu().numpy())

    cm = confusion_matrix(final_targets, final_preds)
    save_confusion_matrix(cm, CLASS_NAMES, OUTPUT_DIR / "confusion_matrix.png")
    print_report(final_targets, final_preds, CLASS_NAMES)

    final_model_path = OUTPUT_DIR / "lstm_classifier.pt"
    torch.save(model.state_dict(), final_model_path)

    print(f"Saved best model to {best_model_path}")
    print(f"Saved final model to {final_model_path}")