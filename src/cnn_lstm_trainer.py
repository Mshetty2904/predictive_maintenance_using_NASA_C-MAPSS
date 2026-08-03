from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import (
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_LSTM_BATCH_SIZE,
    CNN_LSTM_EPOCHS,
    CNN_LSTM_DROPOUT,
    CNN_LSTM_LEARNING_RATE,
    CNN_LSTM_LR_PATIENCE,
    CNN_LSTM_PATIENCE,
    ATTENTION_HEADS,
    ATTENTION_KEY_DIM,
    DENSE_1,
    DENSE_2,
    LSTM_UNITS_2,
    LR_FACTOR,
    MAX_RUL,
    MIN_LR,
    MONITOR,
    RANDOM_STATE,
    VALIDATION_SIZE_BY_DATASET,
)
from src.model_utils import (
    print_cv_fold,
    print_cv_summary,
    print_training_diagnostics,
    save_keras_model_safely,
)
from src.nasa_score import nasa_score
from src.scaler import FeatureScaler


class CNNLSTMTrainer:

    def __init__(self, model_path, scaler_path):
        self.model_path = Path(model_path)
        self.scaler_path = scaler_path
        self.model_path.mkdir(parents=True, exist_ok=True)

    def build_model(self, input_shape):
        """Build the same CNN-BiLSTM-Attention model for every dataset."""
        inputs = Input(shape=input_shape)
        x = Conv1D(
            filters=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            activation="relu",
            padding="same",
            kernel_regularizer=l2(1e-4),
        )(inputs)
        x = BatchNormalization()(x)
        x = Dropout(CNN_LSTM_DROPOUT)(x)

        x = Bidirectional(
            LSTM(
                LSTM_UNITS_2,
                return_sequences=True,
                dropout=CNN_LSTM_DROPOUT,
            )
        )(x)

        attention_output = MultiHeadAttention(
            num_heads=ATTENTION_HEADS,
            key_dim=ATTENTION_KEY_DIM,
            dropout=CNN_LSTM_DROPOUT,
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(
            DENSE_1,
            activation="relu",
            kernel_regularizer=l2(1e-4),
        )(x)
        x = Dropout(CNN_LSTM_DROPOUT)(x)
        x = Dense(
            DENSE_2,
            activation="relu",
            kernel_regularizer=l2(1e-4),
        )(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=CNN_LSTM_LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
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
            model = self.build_model(
                X_train_fold.shape[1:],
            )
            model.fit(
                X_train_fold,
                y_train[train_idx],
                validation_data=(X_valid_fold, y_train[valid_idx]),
                epochs=CNN_LSTM_EPOCHS,
                batch_size=CNN_LSTM_BATCH_SIZE,
                verbose=0,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=CNN_LSTM_PATIENCE,
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

        final_split = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE_BY_DATASET[bundle.dataset_name],
            random_state=RANDOM_STATE,
        )
        fit_idx, valid_idx = next(
            final_split.split(X_train_raw, y_train, groups=bundle.train_groups)
        )
        X_train, X_valid, X_test, _ = scaler.scale_final_data(
            bundle,
            train_idx=fit_idx,
            valid_idx=valid_idx,
        )

        print("\nTraining Final CNN-LSTM Model...")
        final_model = self.build_model(
            X_train.shape[1:],
        )
        model_file = self.model_path / f"{bundle.dataset_name}_cnn_lstm.keras"
        history = final_model.fit(
            X_train,
            y_train[fit_idx],
            validation_data=(X_valid, y_train[valid_idx]),
            epochs=CNN_LSTM_EPOCHS,
            batch_size=CNN_LSTM_BATCH_SIZE,
            callbacks=[
                EarlyStopping(
                    monitor=MONITOR,
                    patience=CNN_LSTM_PATIENCE,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor=MONITOR,
                    factor=LR_FACTOR,
                    patience=CNN_LSTM_LR_PATIENCE,
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
            "CNN-BiLSTM-Attention",
            model=final_model,
            X_train=X_train,
            y_train=y_train[fit_idx],
            X_valid=X_valid,
            y_valid=y_train[valid_idx],
        )
        predictions = final_model.predict(X_test, verbose=0).flatten()
        predictions = np.clip(predictions, 0, MAX_RUL)
        save_keras_model_safely(final_model, model_file)
        self.history = history.history
        return final_model, predictions
