import pandas as pd
import numpy as np

# ============================================================
# STEP 1: FILE PATHS
# ============================================================

train_path = "data/C-MAPSS/train_FD001.txt"
test_path = "data/C-MAPSS/test_FD001.txt"
rul_path = "data/C-MAPSS/RUL_FD001.txt"


# ============================================================
# STEP 2: COLUMN NAMES
# ============================================================

columns = [
    "engine_id",
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]


# ============================================================
# STEP 3: LOAD RAW DATA
# ============================================================

train_df = pd.read_csv(
    train_path,
    sep=r"\s+",
    header=None,
    names=columns
)

test_df = pd.read_csv(
    test_path,
    sep=r"\s+",
    header=None,
    names=columns
)

rul_df = pd.read_csv(
    rul_path,
    sep=r"\s+",
    header=None,
    names=["RUL"]
)


# ============================================================
# STEP 4: BASIC INSPECTION
# ============================================================

print("=" * 60)
print("DATA LOADING")
print("=" * 60)

print("\nTrain shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("RUL shape  :", rul_df.shape)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nFirst 5 rows of TRAIN:")
print(train_df.head())

print("\nFirst 5 rows of TEST:")
print(test_df.head())

print("\nFirst 5 RUL values:")
print(rul_df.head())


# ============================================================
# STEP 5: VERIFY COLUMN COUNT
# ============================================================

print("\n" + "=" * 60)
print("COLUMN VERIFICATION")
print("=" * 60)

print("\nExpected columns:", len(columns))
print("Actual train columns:", len(train_df.columns))
print("Actual test columns :", len(test_df.columns))

assert len(train_df.columns) == 26
assert len(test_df.columns) == 26

print("\nColumn count verification: PASSED")

# ============================================================
# STEP 6: SENSOR EXCLUSION
# ============================================================

all_sensors = [
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

sensors_to_remove = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]

selected_sensors = [
    sensor for sensor in all_sensors
    if sensor not in sensors_to_remove
]

print("\n" + "=" * 60)
print("SENSOR SELECTION")
print("=" * 60)

print("\nTotal sensors before exclusion:",
      len(all_sensors))

print("Sensors removed:",
      sensors_to_remove)

print("Number of sensors removed:",
      len(sensors_to_remove))

print("\nSelected sensors:")
print(selected_sensors)

print("\nNumber of selected sensors:",
      len(selected_sensors))


# Verify sensor selection
assert len(all_sensors) == 21
assert len(sensors_to_remove) == 7
assert len(selected_sensors) == 14

print("\nSensor selection verification: PASSED")

# ============================================================
# STEP 7: MIN-MAX NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("MIN-MAX NORMALIZATION")
print("=" * 60)

# Calculate min and max ONLY from training data
train_min = train_df[selected_sensors].min()
train_max = train_df[selected_sensors].max()

print("\nTraining sensor minimum values:")
print(train_min)

print("\nTraining sensor maximum values:")
print(train_max)


# Check that no sensor has zero range
zero_range_sensors = train_min[train_max == train_min].index.tolist()

assert len(zero_range_sensors) == 0, (
    f"Sensors with zero range found: {zero_range_sensors}"
)

print("\nZero-range sensor check: PASSED")


# Normalize training data
train_df[selected_sensors] = (
    train_df[selected_sensors] - train_min
) / (
    train_max - train_min
)


# Normalize test data using TRAINING min/max
test_df[selected_sensors] = (
    test_df[selected_sensors] - train_min
) / (
    train_max - train_min
)


# ============================================================
# VERIFY NORMALIZATION
# ============================================================

print("\nNormalized TRAIN sensor ranges:")

train_normalized_min = train_df[selected_sensors].min()
train_normalized_max = train_df[selected_sensors].max()

for sensor in selected_sensors:
    print(
        f"{sensor:10s} -> "
        f"min = {train_normalized_min[sensor]:.6f}, "
        f"max = {train_normalized_max[sensor]:.6f}"
    )


print("\nNormalized TEST sensor ranges:")

test_normalized_min = test_df[selected_sensors].min()
test_normalized_max = test_df[selected_sensors].max()

for sensor in selected_sensors:
    print(
        f"{sensor:10s} -> "
        f"min = {test_normalized_min[sensor]:.6f}, "
        f"max = {test_normalized_max[sensor]:.6f}"
    )


# Training data should be exactly within [0, 1]
assert np.allclose(
    train_normalized_min.values,
    0.0
)

assert np.allclose(
    train_normalized_max.values,
    1.0
)

print("\nTraining normalization verification: PASSED")


print("\nFirst 5 normalized TRAIN rows:")
print(train_df[selected_sensors].head())

print("\nFirst 5 normalized TEST rows:")
print(test_df[selected_sensors].head())

# ============================================================
# STEP 8: PIECEWISE LINEAR RUL LABELING
# ============================================================

print("\n" + "=" * 60)
print("PIECEWISE LINEAR RUL LABELING")
print("=" * 60)

RUL_CAP = 130

# Find the maximum cycle reached by each engine
max_cycle_per_engine = train_df.groupby("engine_id")["cycle"].transform("max")

print("\nMaximum cycle for first 10 training engines:")
print(
    train_df.groupby("engine_id")["cycle"]
    .max()
    .head(10)
)


# Calculate raw RUL
train_df["RUL_raw"] = (
    max_cycle_per_engine - train_df["cycle"]
)


# Apply the 130-cycle cap
train_df["RUL"] = train_df["RUL_raw"].clip(
    upper=RUL_CAP
)


# ============================================================
# VERIFY RUL LABELING
# ============================================================

print("\nFirst 20 RUL values for Engine 1:")

engine_1 = train_df[
    train_df["engine_id"] == 1
][
    ["engine_id", "cycle", "RUL_raw", "RUL"]
]

print(engine_1.head(20))

print("\nLast 10 RUL values for Engine 1:")
print(engine_1.tail(10))


# Check final cycle has RUL = 0
final_cycle_rows = train_df.groupby("engine_id")["cycle"].idxmax()

final_cycle_rul = train_df.loc[
    final_cycle_rows, "RUL"
]

assert np.allclose(
    final_cycle_rul.values,
    0
)

print("\nFinal-cycle RUL verification: PASSED")


# Check that no RUL exceeds the cap
assert train_df["RUL"].max() <= RUL_CAP

print("RUL cap verification: PASSED")

print("\nMaximum RUL after capping:",
      train_df["RUL"].max())

print("Minimum RUL:",
      train_df["RUL"].min())

print("\nRUL data type:",
      train_df["RUL"].dtype)

# ============================================================
# STEP 9: SLIDING WINDOW GENERATION
# ============================================================

WINDOW_SIZE = 40
STRIDE = 1


def create_sliding_windows(
    dataframe,
    feature_columns,
    target_column,
    window_size=40,
    stride=1
):
    """
    Create sliding windows independently for each engine.

    Each window contains:
        window_size time steps
        number of selected sensor features

    The label corresponds to the target value
    at the last time step of the window.

    Also stores the engine ID corresponding
    to each generated window.
    """

    X = []
    y = []
    window_engine_ids = []

    skipped_engines = []

    # Process each engine independently
    for engine_id, engine_data in dataframe.groupby(
        "engine_id",
        sort=True
    ):

        # Make sure cycles are in chronological order
        engine_data = engine_data.sort_values("cycle")

        num_cycles = len(engine_data)

        # Skip engines shorter than the window
        if num_cycles < window_size:
            skipped_engines.append(
                (engine_id, num_cycles)
            )
            continue

        # Extract sensor values
        features = engine_data[
            feature_columns
        ].to_numpy(dtype=np.float32)

        # Extract RUL values
        targets = engine_data[
            target_column
        ].to_numpy(dtype=np.float32)

        # Generate windows
        for start in range(
            0,
            num_cycles - window_size + 1,
            stride
        ):

            end = start + window_size

            window = features[start:end]

            # Label = RUL at the LAST time step
            label = targets[end - 1]

            X.append(window)
            y.append(label)

            # Store engine ID for this window
            window_engine_ids.append(engine_id)

    # Convert lists to NumPy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    window_engine_ids = np.asarray(
        window_engine_ids,
        dtype=np.int64
    )

    return (
        X,
        y,
        window_engine_ids,
        skipped_engines
    )


# ============================================================
# CREATE TRAINING WINDOWS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SLIDING WINDOWS")
print("=" * 60)

(
    X_train,
    y_train,
    train_window_engine_ids,
    skipped_train
) = create_sliding_windows(
    dataframe=train_df,
    feature_columns=selected_sensors,
    target_column="RUL",
    window_size=WINDOW_SIZE,
    stride=STRIDE
)


# ============================================================
# VERIFY TRAINING WINDOWS
# ============================================================

print()
print("=" * 60)
print("TRAINING WINDOW ENGINE ID VERIFICATION")
print("=" * 60)

print("Engine ID array shape:", train_window_engine_ids.shape)
print("Engine ID array dtype:", train_window_engine_ids.dtype)

print("First 20 window engine IDs:")
print(train_window_engine_ids[:20])

print("Last 20 window engine IDs:")
print(train_window_engine_ids[-20:])

print("Number of unique engines represented:",
      len(np.unique(train_window_engine_ids)))

print("Unique training engines:")
print(np.unique(train_window_engine_ids))

print("\nWindow size:", WINDOW_SIZE)
print("Stride:", STRIDE)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nX_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)

print("\nNumber of skipped engines:",
      len(skipped_train))

if skipped_train:
    print("Skipped engines:")
    print(skipped_train)
else:
    print("No engines were skipped.")


# ============================================================
# SHAPE VERIFICATION
# ============================================================

assert X_train.ndim == 3
assert X_train.shape[1] == WINDOW_SIZE
assert X_train.shape[2] == len(selected_sensors)

assert y_train.ndim == 1
assert X_train.shape[0] == y_train.shape[0]

print("\nTraining window shape verification: PASSED")


# ============================================================
# VALUE VERIFICATION
# ============================================================

print("\nFirst training window shape:",
      X_train[0].shape)

print("\nFirst training window:")
print(X_train[0])

print("\nFirst training label:",
      y_train[0])

print("\nLast training label:",
      y_train[-1])

# ============================================================
# STEP 10: TEST RUL MAPPING
# ============================================================

print("\n" + "=" * 60)
print("TEST RUL MAPPING")
print("=" * 60)

# RUL file contains one true RUL value for each test engine.
# Row 0 corresponds to engine 1, row 1 to engine 2, etc.

test_engine_ids = sorted(
    test_df["engine_id"].unique()
)

assert len(test_engine_ids) == len(rul_df)

# Create engine_id -> true RUL mapping
test_rul_mapping = dict(
    zip(
        test_engine_ids,
        rul_df["RUL"].values
    )
)

print("\nNumber of test engines:",
      len(test_engine_ids))

print("\nFirst 10 test-engine RUL values:")

for engine_id in test_engine_ids[:10]:
    print(
        f"Engine {engine_id:3d} -> "
        f"True RUL = {test_rul_mapping[engine_id]}"
    )


# ============================================================
# VERIFY RUL MAPPING
# ============================================================

assert set(test_rul_mapping.keys()) == set(
    test_engine_ids
)

assert len(test_rul_mapping) == 100

print("\nTest RUL mapping verification: PASSED")

# ============================================================
# STEP 10B: TEST SLIDING WINDOW GENERATION
# ============================================================

print("\n" + "=" * 60)
print("TEST SLIDING WINDOW GENERATION")
print("=" * 60)


def create_test_windows(
    dataframe,
    feature_columns,
    rul_mapping,
    window_size=40
):
    """
    Create one test window per engine.

    The window corresponds to the final observed cycles
    of each test engine.

    If an engine has fewer than window_size cycles,
    left-pad by repeating its first observation.

    The target is the true RUL provided in RUL_FD001.txt.
    """

    X = []
    y = []

    padded_engines = []

    for engine_id, engine_data in dataframe.groupby(
        "engine_id",
        sort=True
    ):

        # Ensure chronological order
        engine_data = engine_data.sort_values("cycle")

        features = engine_data[
            feature_columns
        ].to_numpy(dtype=np.float32)

        num_cycles = len(features)

        # ----------------------------------------------------
        # CASE 1: Engine has fewer than 40 cycles
        # ----------------------------------------------------

        if num_cycles < window_size:

            padding_size = window_size - num_cycles

            # Repeat the first observation
            padding = np.repeat(
                features[0:1],
                padding_size,
                axis=0
            )

            window = np.vstack(
                [padding, features]
            )

            padded_engines.append(
                (engine_id, num_cycles, padding_size)
            )

        # ----------------------------------------------------
        # CASE 2: Engine has at least 40 cycles
        # ----------------------------------------------------

        else:

            # Take the final 40 observations
            window = features[-window_size:]


        # True RUL comes from RUL_FD001.txt
        true_rul = rul_mapping[engine_id]

        X.append(window)
        y.append(true_rul)


    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    return X, y, padded_engines


# ============================================================
# CREATE TEST WINDOWS
# ============================================================

X_test, y_test, padded_test_engines = create_test_windows(
    dataframe=test_df,
    feature_columns=selected_sensors,
    rul_mapping=test_rul_mapping,
    window_size=WINDOW_SIZE
)


# ============================================================
# TEST WINDOW INFORMATION
# ============================================================

print("\nWindow size:", WINDOW_SIZE)

print("\nX_test shape:",
      X_test.shape)

print("y_test shape:",
      y_test.shape)

print("\nX_test dtype:",
      X_test.dtype)

print("y_test dtype:",
      y_test.dtype)


# ============================================================
# PADDING INFORMATION
# ============================================================

print("\nNumber of padded test engines:",
      len(padded_test_engines))

if padded_test_engines:

    print("\nPadded engines:")
    print(
        "Engine ID | Original cycles | Padding added"
    )

    for engine_id, num_cycles, padding_size in padded_test_engines:

        print(
            f"{engine_id:9d} | "
            f"{num_cycles:15d} | "
            f"{padding_size:13d}"
        )

else:

    print("No test engines required padding.")


# ============================================================
# TEST WINDOW VERIFICATION
# ============================================================

assert X_test.ndim == 3

assert X_test.shape[0] == 100

assert X_test.shape[1] == WINDOW_SIZE

assert X_test.shape[2] == len(selected_sensors)

assert y_test.ndim == 1

assert y_test.shape[0] == 100

assert X_test.shape[0] == y_test.shape[0]

print("\nTest window shape verification: PASSED")


# ============================================================
# TEST RUL VERIFICATION
# ============================================================

print("\nFirst 10 test labels:")

print(y_test[:10])

print("\nExpected first 10 RUL values:")

print(
    rul_df["RUL"].values[:10]
)


assert np.array_equal(
    y_test.astype(int),
    rul_df["RUL"].values
)

print("\nTest RUL label verification: PASSED")


# ============================================================
# SAMPLE TEST WINDOW
# ============================================================

print("\nFirst test window shape:")
print(X_test[0].shape)

print("\nFirst test window - first 3 rows:")
print(X_test[0][:3])

print("\nFirst test window - last 3 rows:")
print(X_test[0][-3:])

print("\nFirst test label:")
print(y_test[0])

# ============================================================
# STEP 6: VERIFY ENGINE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("ENGINE VERIFICATION")
print("=" * 60)

print("\nNumber of training engines:",
      train_df["engine_id"].nunique())

print("Number of test engines:",
      test_df["engine_id"].nunique())

print("\nTraining engines:", 
      train_df["engine_id"].unique())

print("\nTest engines:",
      test_df["engine_id"].unique())


# ============================================================
# STEP 7: VERIFY MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print("\nMissing values in train:",
      train_df.isnull().sum().sum())

print("Missing values in test:",
      test_df.isnull().sum().sum())

print("Missing values in RUL:",
      rul_df.isnull().sum().sum())

assert train_df.isnull().sum().sum() == 0
assert test_df.isnull().sum().sum() == 0
assert rul_df.isnull().sum().sum() == 0

print("\nMissing value verification: PASSED")