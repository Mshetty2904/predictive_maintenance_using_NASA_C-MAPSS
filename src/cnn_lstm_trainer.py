"""
Modified CNNLSTMTrainer with regime-aware conditioning for FD002/FD004.

WHAT CHANGED vs your original file, and why:

1. build_model() optionally takes a second input branch (one-hot regime
   vector) and applies FiLM conditioning to the first conv block's output,
   only when the dataset is in ROBUST_DATASETS and USE_REGIME_FILM is set.
   This gives the model an explicit, undiluted signal for "which operating
   regime is this window in" rather than relying on regime info buried
   among 45 other features going through conv layers.

2. Robust datasets get wider attention (ROBUST_ATTENTION_HEADS/KEY_DIM),
   optional gelu activation, gradient clipping, and their own cosine-restart
   schedule -- all independent from the standard-dataset (FD001/FD003) path,
   which is left as your original config to avoid affecting datasets that
   are currently overfitting rather than underfitting.

3. train() derives regime one-hot vectors and per-sample weights by reading
   column 0 (Regime_ID) of the LAST timestep of each raw window array --
   confirmed against src/window_generator.py::_feature_columns, which places
   Regime_ID first (when present) and Regime_OneHot_* last. This works
   directly on bundle.X_train / bundle.X_test / bundle.final_windows_by_cutoff
   with no separate dataframe lookup, so there's no row-alignment risk: it's
   reading the exact same array the model is trained on.

Everything else (imports, save logic, CV structure, deploy retrain flow)
is unchanged from your original file.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, Activation, Add, BatchNormalization,
                                     Bidirectional, Conv1D, Dense, Dropout,
                                     GlobalAveragePooling1D, Input, Lambda,
                                     LayerNormalization, MultiHeadAttention,
                                     Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import (  # --- new constants, see config_additions_fd002_fd004.py ---
    ACTIVATION, ATTENTION_HEADS, ATTENTION_KEY_DIM, CNN_COSINE_ALPHA,
    CNN_COSINE_FIRST_DECAY_STEPS, CNN_COSINE_M_MUL, CNN_COSINE_T_MUL,
    CNN_FILTERS, CNN_KERNEL_SIZE, CNN_LSTM_BATCH_SIZE, CNN_LSTM_DROPOUT,
    CNN_LSTM_EPOCHS, CNN_LSTM_LEARNING_RATE, CNN_LSTM_LR_PATIENCE,
    CNN_LSTM_PATIENCE, CNN_USE_COSINE_RESTARTS, CNN_USE_RESIDUAL_BLOCK,
    DENSE_1, DENSE_2, HUBER_DELTA, L2_REGULARIZATION, LR_FACTOR, LSTM_UNITS_2,
    MAX_RUL, MIN_LR, MONITOR, RANDOM_STATE, REGIME_ONEHOT_SIZE,
    REGIME_WEIGHT_CAP, ROBUST_ACTIVATION, ROBUST_ATTENTION_HEADS,
    ROBUST_ATTENTION_KEY_DIM, ROBUST_CLIPNORM, ROBUST_CNN_COSINE_ALPHA,
    ROBUST_CNN_COSINE_FIRST_DECAY_STEPS, ROBUST_CNN_COSINE_M_MUL,
    ROBUST_CNN_COSINE_T_MUL, ROBUST_CNN_DROPOUT, ROBUST_CNN_FILTERS,
    ROBUST_CNN_L2, ROBUST_CNN_LEARNING_RATE, ROBUST_CNN_LSTM_UNITS,
    ROBUST_CNN_USE_COSINE_RESTARTS, ROBUST_DATASETS, SAMPLE_WEIGHT_BY_REGIME,
    USE_ADAMW, USE_REGIME_FILM, VALIDATION_SIZE_BY_DATASET, WEIGHT_DECAY)
from src.model_utils import (final_window_split, print_cv_fold,
                             print_cv_summary, print_training_diagnostics,
                             save_keras_model_safely)
from src.nasa_score import nasa_score
from src.scaler import FeatureScaler


class MultiOriginValidationCallback(tf.keras.callbacks.Callback):
    """Monitor the average Huber loss over several truncated engine origins."""

    def __init__(self, validation_sets, delta):
        super().__init__()
        self.validation_sets = validation_sets
        self.delta = delta

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        losses = []
        mses = []
        for X_valid, y_valid in self.validation_sets:
            predictions = self.model.predict(X_valid, verbose=0).reshape(-1)
            errors = predictions - y_valid
            absolute = np.abs(errors)
            huber = np.where(
                absolute <= self.delta,
                0.5 * errors ** 2,
                self.delta * (absolute - 0.5 * self.delta),
            )
            losses.append(float(np.mean(huber)))
            mses.append(float(np.mean(errors ** 2)))
        logs["val_loss"] = float(np.mean(losses))
        logs["val_mse"] = float(np.mean(mses))


class CNNLSTMTrainer:

    def __init__(self, model_path, scaler_path):
        self.model_path = Path(model_path)
        self.scaler_path = scaler_path
        self.model_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Regime helpers
    # ------------------------------------------------------------------
    def _regime_enabled(self, dataset_name, has_regime):
        return (
            has_regime
            and dataset_name in ROBUST_DATASETS
            and (USE_REGIME_FILM or SAMPLE_WEIGHT_BY_REGIME)
        )

    def _build_regime_onehot(self, regime_ids):
        """One-hot encode regime ids with a fixed category set so every
        fold/split produces vectors of the same width, even if a fold
        happens not to contain every regime."""
        categories = pd.CategoricalDtype(categories=range(REGIME_ONEHOT_SIZE))
        encoded = pd.Categorical(regime_ids, dtype=categories)
        onehot = pd.get_dummies(encoded).values.astype("float32")
        return onehot

    def _build_sample_weights(self, regime_ids):
        """Inverse-frequency weights per regime, capped to avoid a rare
        regime dominating the loss."""
        counts = pd.Series(regime_ids).value_counts()
        raw_weights = (counts.max() / counts).clip(upper=REGIME_WEIGHT_CAP)
        weight_lookup = raw_weights.to_dict()
        return np.array([weight_lookup[r] for r in regime_ids], dtype="float32")

    def _get_regime_ids(self, X_windows):
        """Regime ID per window, read directly out of the window array.

        Confirmed from src/window_generator.py::_feature_columns: when
        Regime_ID is present it is placed as column 0, ahead of the sensor
        columns, with Regime_OneHot_* appended as the last REGIME_ONEHOT_SIZE
        columns. Every window in X_train / X_test / final_windows_by_cutoff
        follows this exact layout, so there's no separate dataframe lookup
        or alignment risk -- this reads the same array the model trains on.

        Uses the LAST timestep of each window (index -1), matching how
        window_generator takes the RUL target from the final row of the
        window (`engine.loc[end - 1, "RUL"]`) -- i.e. "regime at the point
        the model is predicting from," which is the right regime label to
        condition on even if the regime shifted earlier within the window.

        IMPORTANT: call this on the RAW (pre-scaling) window array. Regime_ID
        is a raw integer label (0..K-1), not a z-scored sensor value -- do
        not extract it after scaler.fit_transform_pair/_many has run.
        """
        return X_windows[:, -1, 0].astype(int)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def build_model(self, input_shape, dataset_name=None, regime_dim=None):
        """Build the CNN-BiLSTM-Attention model. Robust datasets (FD002/
        FD004) get wider attention, lighter regularization, and an optional
        FiLM regime-conditioning branch; other datasets are unchanged from
        your original architecture."""
        robust = dataset_name in ROBUST_DATASETS
        filters = ROBUST_CNN_FILTERS if robust else CNN_FILTERS
        units = ROBUST_CNN_LSTM_UNITS if robust else LSTM_UNITS_2
        dropout = ROBUST_CNN_DROPOUT if robust else CNN_LSTM_DROPOUT
        learning_rate = ROBUST_CNN_LEARNING_RATE if robust else CNN_LSTM_LEARNING_RATE
        regularizer = l2(ROBUST_CNN_L2 if robust else L2_REGULARIZATION)
        activation = ROBUST_ACTIVATION if robust else ACTIVATION
        attention_heads = ROBUST_ATTENTION_HEADS if robust else ATTENTION_HEADS
        attention_key_dim = ROBUST_ATTENTION_KEY_DIM if robust else ATTENTION_KEY_DIM

        use_film = robust and USE_REGIME_FILM and regime_dim

        inputs = Input(shape=input_shape, name="sequence_input")
        regime_input = None
        if use_film:
            regime_input = Input(shape=(regime_dim,), name="regime_input")

        residual = inputs
        x = Conv1D(
            filters=filters,
            kernel_size=CNN_KERNEL_SIZE,
            activation=None,
            padding="same",
            kernel_regularizer=regularizer,
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv1D(
            filters=filters,
            kernel_size=CNN_KERNEL_SIZE,
            activation=None,
            padding="same",
            kernel_regularizer=regularizer,
        )(x)
        x = BatchNormalization()(x)

        if use_film:
            # gamma initialized around 0 -> scale starts near 1 (via the
            # +1 below), beta initialized around 0 -> shift starts near 0.
            # This keeps the model close to its non-conditioned behavior
            # at init and lets it learn regime-specific adjustments.
            gamma = Dense(filters, name="film_gamma")(regime_input)
            beta = Dense(filters, name="film_beta")(regime_input)
            gamma = Reshape((1, filters))(gamma)
            beta = Reshape((1, filters))(beta)
            x = Lambda(
                lambda t: t[0] * (1.0 + t[1]) + t[2],
                name="film_modulation",
            )([x, gamma, beta])

        if CNN_USE_RESIDUAL_BLOCK:
            residual = Conv1D(
                filters, 1, padding="same", kernel_regularizer=regularizer
            )(residual)
            residual = BatchNormalization()(residual)
            x = Add()([x, residual])
            x = Activation(activation)(x)
        x = Dropout(dropout)(x)

        x = Bidirectional(
            LSTM(
                units,
                return_sequences=True,
                dropout=dropout,
                kernel_regularizer=regularizer,
            )
        )(x)

        attention_output = MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=attention_key_dim,
            dropout=dropout,
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(
            DENSE_1,
            activation=activation,
            kernel_regularizer=regularizer,
        )(x)
        x = Dropout(dropout)(x)
        x = Dense(
            DENSE_2,
            activation=activation,
            kernel_regularizer=regularizer,
        )(x)
        outputs = Dense(1, kernel_regularizer=regularizer)(x)

        model_inputs = [inputs, regime_input] if use_film else inputs
        model = Model(inputs=model_inputs, outputs=outputs)

        use_cosine = ROBUST_CNN_USE_COSINE_RESTARTS if robust else CNN_USE_COSINE_RESTARTS
        if use_cosine:
            first_decay_steps = (
                ROBUST_CNN_COSINE_FIRST_DECAY_STEPS if robust else CNN_COSINE_FIRST_DECAY_STEPS
            )
            t_mul = ROBUST_CNN_COSINE_T_MUL if robust else CNN_COSINE_T_MUL
            m_mul = ROBUST_CNN_COSINE_M_MUL if robust else CNN_COSINE_M_MUL
            alpha = ROBUST_CNN_COSINE_ALPHA if robust else CNN_COSINE_ALPHA
            learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=learning_rate,
                first_decay_steps=first_decay_steps,
                t_mul=t_mul,
                m_mul=m_mul,
                alpha=alpha,
            )

        optimizer_class = tf.keras.optimizers.AdamW if USE_ADAMW else Adam
        optimizer_kwargs = {"learning_rate": learning_rate}
        if USE_ADAMW:
            optimizer_kwargs["weight_decay"] = WEIGHT_DECAY
        if robust and ROBUST_CLIPNORM:
            optimizer_kwargs["clipnorm"] = ROBUST_CLIPNORM
        optimizer = optimizer_class(**optimizer_kwargs)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
            metrics=["mae", "mse"],
        )
        return model

    @staticmethod
    def _as_model_input(X_seq, regime_onehot):
        return [X_seq, regime_onehot] if regime_onehot is not None else X_seq

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, bundle):
        scaler = FeatureScaler(self.scaler_path)
        X_train_raw = bundle.X_train
        y_train = bundle.y_train
        has_regime = "Regime_ID" in bundle.train.columns
        robust = bundle.dataset_name in ROBUST_DATASETS
        regime_active = self._regime_enabled(bundle.dataset_name, has_regime)

        # Extracted from the RAW window array (before scaling) -- see
        # _get_regime_ids docstring. Row order matches X_train_raw / y_train
        # / bundle.train_groups exactly, since all four come out of the same
        # per-engine windowing loop in create_train_windows.
        regime_ids_all = self._get_regime_ids(X_train_raw) if regime_active else None
        regime_dim = REGIME_ONEHOT_SIZE if (regime_active and USE_REGIME_FILM) else None

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

            regime_train_fold = regime_valid_fold = None
            sample_weight_fold = None
            if regime_active:
                fold_regime_ids_train = regime_ids_all[train_idx]
                fold_regime_ids_valid = regime_ids_all[valid_idx]
                if USE_REGIME_FILM:
                    regime_train_fold = self._build_regime_onehot(fold_regime_ids_train)
                    regime_valid_fold = self._build_regime_onehot(fold_regime_ids_valid)
                if SAMPLE_WEIGHT_BY_REGIME:
                    sample_weight_fold = self._build_sample_weights(fold_regime_ids_train)

            model = self.build_model(
                X_train_fold.shape[1:], bundle.dataset_name, regime_dim=regime_dim
            )
            model.fit(
                self._as_model_input(X_train_fold, regime_train_fold),
                y_train[train_idx],
                sample_weight=sample_weight_fold,
                validation_data=(
                    self._as_model_input(X_valid_fold, regime_valid_fold),
                    y_train[valid_idx],
                ),
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
            predictions = model.predict(
                self._as_model_input(X_valid_fold, regime_valid_fold), verbose=0
            ).flatten()
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
        cutoffs = tuple(bundle.final_windows_by_cutoff)
        raw_validation_windows = [
            bundle.final_windows_by_cutoff[cutoff][valid_final_idx]
            for cutoff in cutoffs
        ]
        raw_validation_targets = [
            bundle.final_targets_by_cutoff[cutoff][valid_final_idx]
            for cutoff in cutoffs
        ]
        X_train, *scaled_validation_windows, X_test, _ = scaler.fit_transform_many(
            X_train_raw[fit_idx],
            *raw_validation_windows,
            bundle.X_test,
            has_regime=has_regime,
        )

        regime_train_final = None
        sample_weight_final = None
        regime_valid_finals = [None] * len(scaled_validation_windows)
        if regime_active:
            fit_regime_ids = regime_ids_all[fit_idx]
            if USE_REGIME_FILM:
                regime_train_final = self._build_regime_onehot(fit_regime_ids)
                # raw_validation_windows[i] is itself a raw (pre-scaling)
                # window array pulled from bundle.final_windows_by_cutoff,
                # so the same column-0/last-timestep extraction applies
                # directly -- no separate regime source needed here either.
                regime_valid_finals = [
                    self._build_regime_onehot(self._get_regime_ids(vw))
                    for vw in raw_validation_windows
                ]
            if SAMPLE_WEIGHT_BY_REGIME:
                sample_weight_final = self._build_sample_weights(fit_regime_ids)

        validation_sets = [
            (self._as_model_input(vw, rv), vt)
            for vw, rv, vt in zip(
                scaled_validation_windows, regime_valid_finals, raw_validation_targets
            )
        ]
        primary_index = cutoffs.index(0.8) if 0.8 in cutoffs else 0
        X_valid, y_valid = validation_sets[primary_index]

        print("\nTraining Final CNN-LSTM Model...")
        final_model = self.build_model(
            X_train.shape[1:], bundle.dataset_name, regime_dim=regime_dim
        )
        model_file = self.model_path / f"{bundle.dataset_name}_cnn_lstm.keras"
        final_callbacks = [
            MultiOriginValidationCallback(validation_sets, HUBER_DELTA),
            EarlyStopping(
                monitor=MONITOR,
                patience=CNN_LSTM_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        if not (ROBUST_CNN_USE_COSINE_RESTARTS if robust else CNN_USE_COSINE_RESTARTS):
            final_callbacks.append(
                ReduceLROnPlateau(
                    monitor=MONITOR,
                    factor=LR_FACTOR,
                    patience=CNN_LSTM_LR_PATIENCE,
                    min_lr=MIN_LR,
                    verbose=0,
                )
            )
        history = final_model.fit(
            self._as_model_input(X_train, regime_train_final),
            y_train[fit_idx],
            sample_weight=sample_weight_final,
            validation_data=(X_valid, y_valid),
            epochs=CNN_LSTM_EPOCHS,
            batch_size=CNN_LSTM_BATCH_SIZE,
            callbacks=final_callbacks,
            verbose=0,
        )
        print(f"Best Epoch: {int(np.argmin(history.history['val_loss'])) + 1}")
        print_training_diagnostics(
            history,
            bundle.dataset_name,
            "CNN-BiLSTM-Attention",
            model=final_model,
            X_train=self._as_model_input(X_train, regime_train_final),
            y_train=y_train[fit_idx],
            X_valid=X_valid,
            y_valid=y_valid,
        )
        validation_result = final_model.evaluate(
            X_valid, y_valid, verbose=0, return_dict=True
        )
        self.validation_metrics = {
            "MSE": float(validation_result.get("mse", validation_result["loss"])),
            "MAE": float(validation_result.get("mae", 0.0)),
        }
        valid_predictions = final_model.predict(X_valid, verbose=0).reshape(-1)
        self.validation_metrics.update({
            "RMSE": float(np.sqrt(mean_squared_error(y_valid, valid_predictions))),
            "R2": float(r2_score(y_valid, valid_predictions)),
        })
        selected_epochs = int(np.argmin(history.history["val_loss"]) + 1)
        print("Retraining deployable CNN-LSTM on all training engines...")
        X_all, X_test_deploy, deploy_scaler = scaler.fit_transform_pair(
            bundle.X_train, bundle.X_test, has_regime=has_regime
        )
        joblib.dump(deploy_scaler, self.scaler_path / f"{bundle.dataset_name}_scaler.pkl")

        regime_all_deploy = regime_test_deploy = None
        sample_weight_deploy = None
        if regime_active:
            # bundle.X_train / bundle.X_test are both raw window arrays with
            # the same fixed layout as X_train_raw -- read regime straight
            # out of them, same as everywhere else above.
            all_regime_ids = self._get_regime_ids(bundle.X_train)
            if USE_REGIME_FILM:
                regime_all_deploy = self._build_regime_onehot(all_regime_ids)
                test_regime_ids = self._get_regime_ids(bundle.X_test)
                regime_test_deploy = self._build_regime_onehot(test_regime_ids)
            if SAMPLE_WEIGHT_BY_REGIME:
                sample_weight_deploy = self._build_sample_weights(all_regime_ids)

        deploy_model = self.build_model(
            X_all.shape[1:], bundle.dataset_name, regime_dim=regime_dim
        )
        deploy_model.fit(
            self._as_model_input(X_all, regime_all_deploy),
            y_train,
            sample_weight=sample_weight_deploy,
            epochs=selected_epochs,
            batch_size=CNN_LSTM_BATCH_SIZE,
            verbose=0,
        )
        predictions = deploy_model.predict(
            self._as_model_input(X_test_deploy, regime_test_deploy), verbose=0
        ).flatten()
        predictions = np.clip(predictions, 0, MAX_RUL)
        save_keras_model_safely(deploy_model, model_file)
        self.history = history.history
        return deploy_model, predictions
