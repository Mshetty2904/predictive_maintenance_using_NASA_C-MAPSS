from pathlib import Path

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from src.scaler import FeatureScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
)
from config import (
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DENSE_1,        
    DENSE_2,
    DROPOUT,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    VALIDATION_SPLIT,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LR,
    MONITOR,
)
from tensorflow.keras.optimizers import Adam
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

    def build_model(self, input_shape):

        model = Sequential()

        model.add(
            Input(shape=input_shape)
        )

        model.add(
            LSTM(
                LSTM_UNITS_1,
                return_sequences=True,
            )
        )

        model.add(
            BatchNormalization()
        )

        model.add(
            Dropout(
                DROPOUT,
            )
        )

        model.add(
            LSTM(
                LSTM_UNITS_2,
            )
        )

        model.add(
            BatchNormalization()
        )

        model.add(
            Dropout(
                DROPOUT,
            )
        )

        model.add(
            Dense(
                DENSE_1,
                activation="relu",
            )
        )

        model.add(
            Dense(
                DENSE_2,
                activation="relu",
            )
        )

        model.add(
            Dense(
                1,
            )
        )

        model.compile(
            optimizer=Adam(
                learning_rate=LEARNING_RATE,
            ),
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


        rmse_scores = []
        mae_scores = []
        r2_scores = []

        fold = 1

        gkf = GroupKFold(n_splits=5)

        for train_idx, valid_idx in gkf.split(
            X_train,
            y_train,
            groups=bundle.train_groups,
        ):

            model = self.build_model(
                (
                    X_train.shape[1],
                    X_train.shape[2],
                )
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )

            model.fit(
                X_train[train_idx],
                y_train[train_idx],
                validation_data=(
                    X_train[valid_idx],
                    y_train[valid_idx],
                ),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
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

        model_file = (
            self.model_path
            / f"{bundle.dataset_name}_lstm.keras"
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

        history = final_model.fit(

            X_train,

            y_train,

            validation_split=VALIDATION_SPLIT,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            callbacks=[
                early_stop,
                checkpoint,
                reduce_lr,
            ],

            verbose=1,
        )

        final_model.load_weights(
            model_file,
        )

        predictions = final_model.predict(
            X_test,
            verbose=0,
        ).flatten()

        final_model.save(
            model_file,
        )
        self.history = history.history
        return final_model, predictions

       