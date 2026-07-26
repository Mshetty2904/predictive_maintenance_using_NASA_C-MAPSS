from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.scaler import FeatureScaler


class LSTMTrainer:

    def __init__(
        self,
        model_path,
        scaler_path,
    ):

        self.model_path = Path(model_path)
        self.scaler_path = scaler_path

        self.model_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build_model(
        self,
        input_shape,
    ):

        model = Sequential()

        model.add(
            LSTM(
                64,
                input_shape=input_shape,
            )
        )

        model.add(
            Dropout(
                0.2,
            )
        )

        model.add(
            Dense(
                32,
                activation="relu",
            )
        )

        model.add(
            Dense(
                1,
            )
        )

        model.compile(
            optimizer=Adam(),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train(
        self,
        bundle,
    ):

        scaler = FeatureScaler(
            self.scaler_path,
        )

        X_train, X_test = scaler.scale(
            bundle,
        )

        y_train = bundle.y_train

        print("\nRunning 5-Fold Cross Validation...")

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        rmse_scores = []
        mae_scores = []
        r2_scores = []

        fold = 1

        for train_idx, valid_idx in kfold.split(X_train):

            model = self.build_model(
                (
                    X_train.shape[1],
                    X_train.shape[2],
                )
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )

            model.fit(
                X_train[train_idx],
                y_train[train_idx],
                validation_data=(
                    X_train[valid_idx],
                    y_train[valid_idx],
                ),
                epochs=50,
                batch_size=64,
                verbose=0,
                callbacks=[early_stop],
            )

            predictions = model.predict(
                X_train[valid_idx],
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

            print(
                f"Fold {fold}: "
                f"RMSE={rmse:.3f} "
                f"MAE={mae:.3f} "
                f"R2={r2:.3f}"
            )

            fold += 1

        print("\nAverage Cross Validation Performance")

        print(
            f"RMSE : {np.mean(rmse_scores):.3f}"
        )

        print(
            f"MAE  : {np.mean(mae_scores):.3f}"
        )

        print(
            f"R2   : {np.mean(r2_scores):.3f}"
        )

        print("\nTraining Final Model...")

        final_model = self.build_model(
            (
                X_train.shape[1],
                X_train.shape[2],
            )
        )

        final_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=64,
            verbose=0,
        )

        predictions = final_model.predict(
            X_test,
            verbose=0,
        ).flatten()

        model_file = (
            self.model_path
            / f"{bundle.dataset_name}_lstm.keras"
        )

        final_model.save(
            model_file,
        )

        return final_model, predictions