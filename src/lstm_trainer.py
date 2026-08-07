from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import (
    BATCH_SIZE,
    DENSE_1,
    DENSE_2,
    LSTM_DROPOUT,
    EPOCHS,
    LSTM_LEARNING_RATE,
    LR_FACTOR,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    MAX_RUL,
    MIN_LR,
    MONITOR,
    LSTM_PATIENCE,
    VALIDATION_SIZE_BY_DATASET,
    RANDOM_STATE,
    L2_REGULARIZATION,
    HUBER_DELTA,
    ROBUST_DATASETS,
    ROBUST_LSTM_UNITS,
    ROBUST_DENSE_1,
    ROBUST_DENSE_2,
    ROBUST_DROPOUT,
    ROBUST_LEARNING_RATE,
    ROBUST_L2,
    USE_ADAMW,
    WEIGHT_DECAY,
    ACTIVATION,
)
from src.model_utils import (
    print_cv_fold,
    print_cv_summary,
    print_training_diagnostics,
    final_window_split,
    save_keras_model_safely,
)
from src.scaler import FeatureScaler
from src.nasa_score import nasa_score


class LSTMTrainer:

    def __init__(self, model_path, scaler_path):
        self.model_path = Path(model_path)
        self.scaler_path = scaler_path
        self.model_path.mkdir(parents=True, exist_ok=True)

    def build_model(self, input_shape, dataset_name=None):
        robust = dataset_name in ROBUST_DATASETS
        units = ROBUST_LSTM_UNITS if robust else LSTM_UNITS_1
        dense_1 = ROBUST_DENSE_1 if robust else DENSE_1
        dense_2 = ROBUST_DENSE_2 if robust else DENSE_2
        dropout = ROBUST_DROPOUT if robust else LSTM_DROPOUT
        learning_rate = ROBUST_LEARNING_RATE if robust else LSTM_LEARNING_RATE
        regularizer = l2(ROBUST_L2 if robust else L2_REGULARIZATION)
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(units, kernel_regularizer=regularizer),
                Dropout(dropout),
                Dense(dense_1, activation=ACTIVATION, kernel_regularizer=regularizer),
                Dropout(dropout),
                Dense(dense_2, activation=ACTIVATION, kernel_regularizer=regularizer),
                Dense(1, kernel_regularizer=regularizer),
            ]
        )
        optimizer_class = tf.keras.optimizers.AdamW if USE_ADAMW else Adam
        optimizer = optimizer_class(
            learning_rate=learning_rate,
            **({"weight_decay": WEIGHT_DECAY} if USE_ADAMW else {}),
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
            metrics=["mae", "mse"],
        )
        return model

    def train(self, bundle):
        scaler = FeatureScaler(self.scaler_path)
        X_train_raw = bundle.X_train
        y_train = bundle.y_train
        has_regime = "Regime_ID" in bundle.train.columns

        print("\nRunning 5-Fold Engine-wise Cross Validation...")
        rmse_scores, mae_scores, r2_scores, nasa_scores = [], [], [], []
        group_kfold = GroupKFold(n_splits=5)

        for fold, (train_idx, valid_idx) in enumerate(
            group_kfold.split(X_train_raw, y_train, groups=bundle.train_groups),
            start=1,
        ):
            X_train_fold, X_valid_fold, _ = scaler.fit_transform_pair(
                X_train_raw[train_idx],
                X_train_raw[valid_idx],
                has_regime=has_regime,
            )
            model = self.build_model(X_train_fold.shape[1:], bundle.dataset_name)
            model.fit(
                X_train_fold,
                y_train[train_idx],
                validation_data=(X_valid_fold, y_train[valid_idx]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=LSTM_PATIENCE,
                        restore_best_weights=True,
                        verbose=0,
                    )
                ],
            )
            predictions = model.predict(X_valid_fold, verbose=0).flatten()
            predictions = np.clip(predictions, 0, MAX_RUL)
            rmse = np.sqrt(mean_squared_error(y_train[valid_idx], predictions))
            mae = mean_absolute_error(y_train[valid_idx], predictions)
            r2 = r2_score(y_train[valid_idx], predictions)
            score = nasa_score(y_train[valid_idx], predictions)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            nasa_scores.append(score)
            print_cv_fold(fold, rmse, mae, r2, score)

        print_cv_summary(rmse_scores, mae_scores, r2_scores, nasa_scores)

        fit_idx, valid_final_idx = final_window_split(
            bundle,
            VALIDATION_SIZE_BY_DATASET[bundle.dataset_name],
            RANDOM_STATE,
        )
        X_train, X_valid, X_test, _ = scaler.fit_transform_many(
            X_train_raw[fit_idx],
            bundle.X_final[valid_final_idx],
            bundle.X_test,
            has_regime=has_regime,
        )

        print("\nTraining Final LSTM Model...")
        final_model = self.build_model(X_train.shape[1:], bundle.dataset_name)
        model_file = self.model_path / f"{bundle.dataset_name}_lstm.keras"
        history = final_model.fit(
            X_train,
            y_train[fit_idx],
            validation_data=(X_valid, bundle.y_final[valid_final_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(
                    monitor=MONITOR,
                    patience=LSTM_PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor=MONITOR,
                    factor=LR_FACTOR,
                    patience=LSTM_PATIENCE // 2,
                    min_lr=MIN_LR,
                    verbose=0,
                ),
            ],
            verbose=0,
        )
        print(f"Best Epoch: {int(np.argmin(history.history['val_loss'])) + 1}")
        print_training_diagnostics(
            history,
            bundle.dataset_name,
            "LSTM",
            model=final_model,
            X_train=X_train,
            y_train=y_train[fit_idx],
            X_valid=X_valid,
            y_valid=bundle.y_final[valid_final_idx],
        )
        validation_result = final_model.evaluate(
            X_valid, bundle.y_final[valid_final_idx], verbose=0, return_dict=True
        )
        self.validation_metrics = {
            "MSE": float(validation_result.get("mse", validation_result["loss"])),
            "MAE": float(validation_result.get("mae", 0.0)),
        }
        valid_predictions = final_model.predict(X_valid, verbose=0).reshape(-1)
        valid_targets = bundle.y_final[valid_final_idx]
        self.validation_metrics.update({
            "RMSE": float(np.sqrt(mean_squared_error(valid_targets, valid_predictions))),
            "R2": float(r2_score(valid_targets, valid_predictions)),
        })
        selected_epochs = int(np.argmin(history.history["val_loss"]) + 1)
        print("Retraining deployable LSTM on all training engines...")
        X_all, X_test_deploy, deploy_scaler = scaler.fit_transform_pair(
            bundle.X_train, bundle.X_test, has_regime=has_regime
        )
        joblib.dump(deploy_scaler, self.scaler_path / f"{bundle.dataset_name}_scaler.pkl")
        deploy_model = self.build_model(X_all.shape[1:], bundle.dataset_name)
        deploy_model.fit(X_all, y_train, epochs=selected_epochs, batch_size=BATCH_SIZE, verbose=0)
        predictions = deploy_model.predict(X_test_deploy, verbose=0).flatten()
        predictions = np.clip(predictions, 0, MAX_RUL)
        save_keras_model_safely(deploy_model, model_file)
        self.history = history.history
        return deploy_model, predictions
