from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import (
    CNN_FILTERS,
    CNN_LSTM_BATCH_SIZE,
    CNN_LSTM_EPOCHS,
    DENSE_1,
    DENSE_2,
    DROPOUT,
    LEARNING_RATE,
    LR_FACTOR,
    LR_PATIENCE,
    LSTM_UNITS_2,
    MIN_LR,
    MONITOR,
    PATIENCE,
    POOL_SIZE,
    VALIDATION_SPLIT,
)
from src.model_utils import print_cv_fold, print_cv_summary
from src.scaler import FeatureScaler


class EpochLogger(Callback):

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        values = []

        for name in [
            "loss",
            "mae",
            "val_loss",
            "val_mae",
        ]:

            if name in logs:
                values.append(
                    f"{name}={logs[name]:.5f}"
                )

        learning_rate = K.get_value(
            self.model.optimizer.learning_rate
        )

        values.append(
            f"lr={learning_rate:.7f}"
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            + " | ".join(values)
        )


class CNNLSTMTrainer:

    def __init__(self, model_path, scaler_path):

        self.model_path = Path(model_path)
        self.scaler_path = scaler_path

        self.model_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build_model(self, input_shape):

        model = Sequential(
            [
                Input(shape=input_shape),

                Conv1D(
                    filters=CNN_FILTERS,
                    kernel_size=5,
                    activation="relu",
                    padding="same",
                ),

                BatchNormalization(),

                Conv1D(
                    filters=CNN_FILTERS,
                    kernel_size=3,
                    activation="relu",
                    padding="same",
                ),

                BatchNormalization(),

                MaxPooling1D(
                    pool_size=POOL_SIZE,
                ),

                Dropout(DROPOUT),

                LSTM(LSTM_UNITS_2),

                Dropout(DROPOUT),

                Dense(
                    DENSE_1,
                    activation="relu",
                ),

                Dense(
                    DENSE_2,
                    activation="relu",
                ),

                Dense(1),
            ]
        )

        model.compile(
            optimizer=Adam(
                learning_rate=LEARNING_RATE,
            ),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train(self, bundle):

        scaler = FeatureScaler(
            self.scaler_path,
        )

        X_train_raw = bundle.X_train
        y_train = bundle.y_train

        print(
            "\nRunning 5-Fold Engine-wise "
            "Cross Validation..."
        )

        group_kfold = GroupKFold(
            n_splits=5,
        )

        rmse_scores = []
        mae_scores = []
        r2_scores = []

        for fold, (train_idx, valid_idx) in enumerate(
            group_kfold.split(
                X_train_raw,
                y_train,
                groups=bundle.train_groups,
            ),
            start=1,
        ):

            X_train_fold, X_valid_fold, _ = (
                scaler.fit_transform_pair(
                    X_train_raw[train_idx],
                    X_train_raw[valid_idx],
                )
            )

            model = self.build_model(
                (
                    X_train_fold.shape[1],
                    X_train_fold.shape[2],
                )
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )

            model.fit(
                X_train_fold,
                y_train[train_idx],
                validation_data=(
                    X_valid_fold,
                    y_train[valid_idx],
                ),
                epochs=CNN_LSTM_EPOCHS,
                batch_size=CNN_LSTM_BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop],
            )

            predictions = model.predict(
                X_valid_fold,
                verbose=0,
            ).flatten()

            rmse = np.sqrt(
                mean_squared_error(
                    y_train[valid_idx],
                    predictions,
                )
            )

            mae = mean_absolute_error(
                y_train[valid_idx],
                predictions,
            )

            r2 = r2_score(
                y_train[valid_idx],
                predictions,
            )

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print_cv_fold(
                fold,
                rmse,
                mae,
                r2,
            )

        print_cv_summary(
            rmse_scores,
            mae_scores,
            r2_scores,
        )

        X_train, X_test = scaler.scale_final_data(
            bundle,
        )

        print("\nTraining Final CNN-LSTM Model...")
        print("Epoch details are shown below.")

        final_model = self.build_model(
            (
                X_train.shape[1],
                X_train.shape[2],
            )
        )

        model_file = (
            self.model_path
            / f"{bundle.dataset_name}_cnn_lstm.keras"
        )

        early_stop = EarlyStopping(
            monitor=MONITOR,
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

        checkpoint = ModelCheckpoint(
            filepath=model_file,
            monitor=MONITOR,
            save_best_only=True,
            verbose=1,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor=MONITOR,
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        )

        epoch_logger = EpochLogger()

        history = final_model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=CNN_LSTM_EPOCHS,
            batch_size=CNN_LSTM_BATCH_SIZE,
            callbacks=[
                checkpoint,
                reduce_lr,
                epoch_logger,
                early_stop,
            ],
            verbose=0,
        )

        best_epoch = (
            int(
                np.argmin(
                    history.history["val_loss"]
                )
            )
            + 1
        )

        best_val_loss = min(
            history.history["val_loss"]
        )

        print(
            f"\nBest Epoch: {best_epoch}"
        )

        print(
            f"Best Validation Loss: "
            f"{best_val_loss:.5f}"
        )

        if early_stop.stopped_epoch > 0:
            print(
                f"Early stopping activated at epoch "
                f"{early_stop.stopped_epoch + 1}."
            )
        else:
            print(
                "Early stopping was not activated."
            )

        final_model.load_weights(
            model_file,
        )

        predictions = final_model.predict(
            X_test,
            verbose=0,
        ).flatten()

        final_model.save(model_file)

        self.history = history.history

        return final_model, predictions