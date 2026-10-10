import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from preprocessing import X_train, y_train, X_test, y_test


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_SIZE = 14
HIDDEN_SIZE = 45
NUM_LAYERS = 2
OUTPUT_SIZE = 1


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("PHASE 2 - PLAIN BiLSTM BASELINE")
print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# PREPROCESSED DATA VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSED DATA VERIFICATION")
print("=" * 60)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)

print("X_test dtype :", X_test.dtype)
print("y_test dtype :", y_test.dtype)


# Expected shapes from preprocessing.py
assert X_train.shape == (16731, 40, 14), \
    f"Unexpected X_train shape: {X_train.shape}"

assert y_train.shape == (16731,), \
    f"Unexpected y_train shape: {y_train.shape}"

assert X_test.shape == (100, 40, 14), \
    f"Unexpected X_test shape: {X_test.shape}"

assert y_test.shape == (100,), \
    f"Unexpected y_test shape: {y_test.shape}"

assert X_train.dtype == np.float32
assert y_train.dtype == np.float32

assert X_test.dtype == np.float32
assert y_test.dtype == np.float32

print("\nPREPROCESSED DATA VERIFICATION: PASSED")

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / VALIDATION SPLIT")
print("=" * 60)

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

# Reproducible random generator
rng = np.random.default_rng(RANDOM_SEED)

num_samples = X_train.shape[0]

# Create shuffled indices
indices = rng.permutation(num_samples)

# Calculate split point
split_index = int(num_samples * (1 - VALIDATION_RATIO))

train_indices = indices[:split_index]
val_indices = indices[split_index:]

# Create train split
X_train_split = X_train[train_indices]
y_train_split = y_train[train_indices]

# Create validation split
X_val = X_train[val_indices]
y_val = y_train[val_indices]


# ============================================================
# SPLIT VERIFICATION
# ============================================================

print("Total training samples :", num_samples)
print("Training samples       :", len(X_train_split))
print("Validation samples     :", len(X_val))

print("\nX_train_split shape:", X_train_split.shape)
print("y_train_split shape:", y_train_split.shape)

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# Verify expected sizes
expected_train_size = int(num_samples * 0.90)
expected_val_size = num_samples - expected_train_size

assert len(X_train_split) == expected_train_size
assert len(y_train_split) == expected_train_size

assert len(X_val) == expected_val_size
assert len(y_val) == expected_val_size

# Verify shapes
assert X_train_split.shape[1:] == (40, 14)
assert X_val.shape[1:] == (40, 14)

assert y_train_split.ndim == 1
assert y_val.ndim == 1

# Verify no samples were lost
assert (
    len(X_train_split) + len(X_val)
    == len(X_train)
)

print("\nTRAIN / VALIDATION SPLIT: PASSED")

# ============================================================
# PYTORCH DATASET
# ============================================================

class RULDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        # Basic shape checks
        assert self.X.ndim == 3, (
            f"Expected X to be 3D, got {self.X.ndim}D"
        )

        assert self.y.ndim == 1, (
            f"Expected y to be 1D, got {self.y.ndim}D"
        )

        assert self.X.shape[0] == self.y.shape[0], (
            f"X and y sample counts do not match: "
            f"{self.X.shape[0]} vs {self.y.shape[0]}"
        )

        assert self.X.dtype == torch.float32
        assert self.y.dtype == torch.float32

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# ============================================================
# DATALOADER CONFIGURATION
# ============================================================

BATCH_SIZE = 32

train_dataset = RULDataset(
    X_train_split,
    y_train_split
)

val_dataset = RULDataset(
    X_val,
    y_val
)

test_dataset = RULDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# DATALOADER SHAPE SANITY TEST
# ============================================================

print("\n" + "=" * 60)
print("DATALOADER SHAPE SANITY TEST")
print("=" * 60)

# Get one real training batch
train_batch_X, train_batch_y = next(iter(train_loader))

print("Training batch X shape:", train_batch_X.shape)
print("Training batch y shape:", train_batch_y.shape)

print("Training batch X dtype:", train_batch_X.dtype)
print("Training batch y dtype:", train_batch_y.dtype)


# Expected:
# X -> (32, 40, 14)
# y -> (32,)

assert train_batch_X.shape == (32, 40, 14), (
    f"Expected training X batch shape (32, 40, 14), "
    f"got {train_batch_X.shape}"
)

assert train_batch_y.shape == (32,), (
    f"Expected training y batch shape (32,), "
    f"got {train_batch_y.shape}"
)

assert train_batch_X.dtype == torch.float32
assert train_batch_y.dtype == torch.float32


# ------------------------------------------------------------
# Validation batch
# ------------------------------------------------------------

val_batch_X, val_batch_y = next(iter(val_loader))

print("\nValidation batch X shape:", val_batch_X.shape)
print("Validation batch y shape:", val_batch_y.shape)

assert val_batch_X.shape == (32, 40, 14)
assert val_batch_y.shape == (32,)


# ------------------------------------------------------------
# Test batch
# ------------------------------------------------------------

test_batch_X, test_batch_y = next(iter(test_loader))

print("\nTest batch X shape:", test_batch_X.shape)
print("Test batch y shape:", test_batch_y.shape)

assert test_batch_X.shape == (32, 40, 14)
assert test_batch_y.shape == (32,)


print("\nDATALOADER SHAPE VERIFICATION: PASSED")


# ============================================================
# BiLSTM MODEL
# ============================================================

class BiLSTMModel(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 2-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM output size = hidden_size * 2
        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, x, debug=False):

        # ----------------------------------------------------
        # Input shape
        # ----------------------------------------------------
        assert x.ndim == 3, (
            f"Expected 3D input, got {x.ndim}D"
        )

        assert x.shape[2] == INPUT_SIZE, (
            f"Expected {INPUT_SIZE} features, "
            f"got {x.shape[2]}"
        )

        if debug:
            print(f"Input to BiLSTM: {x.shape}")

        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------
        lstm_out, _ = self.lstm(x)

        # Expected:
        # (batch_size, sequence_length, hidden_size * 2)
        if debug:
            print(f"BiLSTM output: {lstm_out.shape}")

        expected_features = HIDDEN_SIZE * 2

        assert lstm_out.shape[2] == expected_features, (
            f"Expected BiLSTM feature size {expected_features}, "
            f"got {lstm_out.shape[2]}"
        )

        # ----------------------------------------------------
        # Take final time step
        # ----------------------------------------------------
        final_output = lstm_out[:, -1, :]

        # Expected:
        # (batch_size, hidden_size * 2)
        if debug:
            print(f"Final timestep output: {final_output.shape}")

        assert final_output.shape[1] == expected_features, (
            f"Expected final feature size {expected_features}, "
            f"got {final_output.shape[1]}"
        )

        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------
        output = self.fc(final_output)

        # Expected:
        # (batch_size, 1)
        if debug:
            print(f"Final FC output: {output.shape}")

        assert output.shape[1] == OUTPUT_SIZE, (
            f"Expected output size {OUTPUT_SIZE}, "
            f"got {output.shape[1]}"
        )

        return output


# ============================================================
# REAL DATA -> MODEL SHAPE SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("REAL DATA -> MODEL SHAPE SANITY TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BiLSTMModel().to(device)

    print("\nModel created successfully.")
    print(f"Model device: {device}")

    # --------------------------------------------------------
    # Get one REAL batch from DataLoader
    # --------------------------------------------------------

    real_X_batch, real_y_batch = next(iter(train_loader))

    print("\nReal DataLoader batch:")
    print("X shape:", real_X_batch.shape)
    print("y shape:", real_y_batch.shape)

    # --------------------------------------------------------
    # Verify batch before sending to model
    # --------------------------------------------------------

    assert real_X_batch.shape == (32, 40, 14), (
        f"Expected X batch shape (32, 40, 14), "
        f"got {real_X_batch.shape}"
    )

    assert real_y_batch.shape == (32,), (
        f"Expected y batch shape (32,), "
        f"got {real_y_batch.shape}"
    )

    # --------------------------------------------------------
    # Move batch to device
    # --------------------------------------------------------

    real_X_batch = real_X_batch.to(device)
    real_y_batch = real_y_batch.to(device)

    print("\nAfter moving to device:")
    print("X device:", real_X_batch.device)
    print("y device:", real_y_batch.device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        predictions = model(real_X_batch, debug=True)

    # --------------------------------------------------------
    # Verify model output
    # --------------------------------------------------------

    print("\nReal model prediction shape:")
    print(predictions.shape)

    assert predictions.shape == (32, 1), (
        f"Expected prediction shape (32, 1), "
        f"got {predictions.shape}"
    )

    # --------------------------------------------------------
    # Convert prediction to 1D for comparison
    # --------------------------------------------------------

    predictions_1d = predictions.squeeze(1)

    print("\nAfter squeeze:")
    print("Prediction shape:", predictions_1d.shape)
    print("Target shape    :", real_y_batch.shape)

    assert predictions_1d.shape == real_y_batch.shape

    # --------------------------------------------------------
    # Display a few values
    # --------------------------------------------------------

    print("\nFirst 5 predictions:")
    print(predictions_1d[:5].detach().cpu().numpy())

    print("\nFirst 5 true RUL values:")
    print(real_y_batch[:5].detach().cpu().numpy())

    print("\n" + "=" * 60)
    print("REAL DATA -> MODEL VERIFICATION: PASSED")
    print("=" * 60)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50
LEARNING_RATE = 0.001

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 TRAINING")
print("=" * 60)

print(f"Epochs       : {EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss         : MSE")
print(f"Optimizer    : Adam")
print(f"Device       : {device}")


train_losses = []
val_losses = []

# ============================================================
# BEST MODEL CHECKPOINT
# ============================================================

best_val_loss = float("inf")
best_epoch = 0

best_model_path = "models/best_bilstm_fd001.pt"
os.makedirs("models", exist_ok=True)


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0
    train_samples = 0

    for X_batch, y_batch in train_loader:

        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)

        # Remove final dimension:
        # (batch_size, 1) -> (batch_size,)
        predictions = predictions.squeeze(1)

        # Shape verification
        assert predictions.shape == y_batch.shape, (
            f"Prediction/target shape mismatch: "
            f"{predictions.shape} vs {y_batch.shape}"
        )

        # Calculate MSE
        loss = criterion(
            predictions,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate loss
        batch_size_actual = X_batch.size(0)

        running_train_loss += (
            loss.item() * batch_size_actual
        )

        train_samples += batch_size_actual

    # Average training loss
    epoch_train_loss = (
        running_train_loss / train_samples
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0
    val_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            predictions = predictions.squeeze(1)

            assert predictions.shape == y_batch.shape

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size_actual = X_batch.size(0)

            running_val_loss += (
                loss.item() * batch_size_actual
            )

            val_samples += batch_size_actual

    # Average validation loss
    epoch_val_loss = (
        running_val_loss / val_samples
    )


    # --------------------------------------------------------
    # STORE LOSSES
    # --------------------------------------------------------

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)


    # --------------------------------------------------------
    # PRINT EPOCH RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Val Loss: {epoch_val_loss:.4f}"
    )
        # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"  -> Best model saved "
            f"(Epoch {epoch}, "
            f"Val Loss: {best_val_loss:.4f})"
        )


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Checkpoint:", best_model_path)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)

model.to(device)
model.eval()

print("\nBest model loaded successfully.")

# ============================================================
# PHASE 2 TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 TEST SET EVALUATION")
print("=" * 60)

model.eval()

test_predictions = []
test_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Model prediction
        predictions = model(X_batch)

        # Convert (batch_size, 1) -> (batch_size,)
        predictions = predictions.squeeze(1)

        # Shape verification
        assert predictions.shape == y_batch.shape, (
            f"Prediction/target shape mismatch: "
            f"{predictions.shape} vs {y_batch.shape}"
        )

        # Store predictions and targets
        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_targets.extend(
            y_batch.cpu().numpy()
        )


# Convert to NumPy arrays
test_predictions = np.asarray(
    test_predictions,
    dtype=np.float32
)

test_targets = np.asarray(
    test_targets,
    dtype=np.float32
)


# ============================================================
# TEST PREDICTION SHAPE VERIFICATION
# ============================================================

print("\nTest predictions shape:",
      test_predictions.shape)

print("Test targets shape   :",
      test_targets.shape)

assert test_predictions.shape == (100,)
assert test_targets.shape == (100,)

print("\nTest prediction shape verification: PASSED")


# ============================================================
# CALCULATE MAE
# ============================================================

mae = np.mean(
    np.abs(
        test_predictions - test_targets
    )
)


# ============================================================
# CALCULATE RMSE
# ============================================================

rmse = np.sqrt(
    np.mean(
        (test_predictions - test_targets) ** 2
    )
)

# ============================================================
# CALCULATE R² SCORE
# ============================================================

ss_res = np.sum(
    (test_targets - test_predictions) ** 2
)

ss_tot = np.sum(
    (test_targets - np.mean(test_targets)) ** 2
)

r2 = 1 - (ss_res / ss_tot)


# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Best validation epoch : {best_epoch}")
print(f"Best validation MSE   : {best_val_loss:.4f}")

print(f"\nTest MAE              : {mae:.4f}")
print(f"Test RMSE             : {rmse:.4f}")
print(f"Test R²               : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE TEST PREDICTIONS")
print("=" * 60)

print(
    f"{'Engine':>8} "
    f"{'True RUL':>12} "
    f"{'Predicted RUL':>16} "
    f"{'Error':>12}"
)

for i in range(10):

    error = (
        test_predictions[i]
        - test_targets[i]
    )

    print(
        f"{i + 1:8d} "
        f"{test_targets[i]:12.2f} "
        f"{test_predictions[i]:16.2f} "
        f"{error:12.2f}"
    )


# ============================================================
# EVALUATION COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 TEST EVALUATION COMPLETED")
print("=" * 60)