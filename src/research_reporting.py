"""Research-grade EDA, model comparison, SHAP, PDP, and ICE reporting."""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from config import MAX_RUL, REGIME_ONEHOT_SIZE, SHAP_MODEL_BY_DATASET, USE_REGIME_FILM


class ResearchReporter:
    def __init__(self, output_root, scaler_root, window_size, max_rul):
        self.root = Path(output_root) / "research"
        self.scaler_root = Path(scaler_root)
        self.window_size = window_size
        self.max_rul = max_rul
        self._make_dirs()

    def _make_dirs(self):
        for name in ("eda", "models", "explainability", "tables"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sensor_columns(data):
        return [c for c in data.columns if c.startswith("Sensor_")]

    @staticmethod
    def _feature_columns(bundle):
        columns = []
        if "Regime_ID" in bundle.train.columns:
            columns.append("Regime_ID")
        columns.extend(ResearchReporter._sensor_columns(bundle.train))
        columns.extend(sorted(c for c in bundle.train.columns if c.startswith("Regime_OneHot_")))
        return columns

    def _dataset_dir(self, dataset):
        path = self.root / "datasets" / dataset
        for name in ("eda", "models", "explainability", "tables"):
            (path / name).mkdir(parents=True, exist_ok=True)
        return path

    def _savefig(self, path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_eda(self, bundle):
        dataset_dir = self._dataset_dir(bundle.dataset_name)
        train = bundle.train.copy()
        sensors = self._sensor_columns(train)
        eda_dir = dataset_dir / "eda"

        # 1. Sensor trend analysis: representative engines and readable facets.
        engine_ids = train.groupby("Engine_ID")["Cycle"].max().sort_values()
        chosen = [engine_ids.index[0], engine_ids.index[len(engine_ids)//2], engine_ids.index[-1]]
        selected_sensors = sensors[:min(6, len(sensors))]
        fig, axes = plt.subplots(len(selected_sensors), 1, figsize=(12, 2.4 * len(selected_sensors)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, sensor in zip(axes, selected_sensors):
            for engine_id in chosen:
                part = train[train["Engine_ID"] == engine_id].sort_values("Cycle")
                ax.plot(part["Cycle"], part[sensor], label=f"Engine {engine_id}")
            ax.set_ylabel(sensor)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Cycle")
        axes[0].legend(ncol=3)
        self._savefig(eda_dir / "sensor_trends_vs_cycle.png")

        # 2. Correlation heatmap.
        plt.figure(figsize=(12, 10))
        sns.heatmap(train[sensors].corr(), cmap="vlag", center=0, square=True)
        plt.title(f"{bundle.dataset_name}: Sensor Correlation Heatmap")
        self._savefig(eda_dir / "sensor_correlation_heatmap.png")

        # 3 and 16. Piecewise RUL distribution and cycle-gap summary.
        plt.figure(figsize=(9, 5))
        sns.histplot(train["RUL"], bins=30, kde=True)
        plt.xlabel("Piecewise RUL")
        plt.ylabel("Number of observations")
        plt.title(f"{bundle.dataset_name}: RUL Distribution After Cycle Gap")
        self._savefig(eda_dir / "rul_distribution_after_cycle_gap.png")

        # 17. Engine degradation curves.
        plt.figure(figsize=(11, 6))
        for engine_id in chosen:
            part = train[train["Engine_ID"] == engine_id].sort_values("Cycle")
            plt.plot(part["Cycle"], part["RUL"], label=f"Engine {engine_id}")
        plt.xlabel("Cycle")
        plt.ylabel("RUL")
        plt.title(f"{bundle.dataset_name}: Engine Degradation Curves")
        plt.legend()
        plt.grid(alpha=0.25)
        self._savefig(eda_dir / "engine_degradation_curves.png")

        # 15. Data summary table.
        summary = pd.DataFrame([{
            "Dataset": bundle.dataset_name,
            "Training engines": train["Engine_ID"].nunique(),
            "Test engines": bundle.test["Engine_ID"].nunique(),
            "Sensors": len(sensors),
            "Operational settings": len([c for c in train.columns if c.startswith("Setting_")]),
            "Window size": self.window_size,
            "MAX_RUL": self.max_rul,
            "Training rows": len(train),
            "Training windows": len(bundle.X_train),
        }])
        summary.to_csv(dataset_dir / "tables" / "data_summary.csv", index=False)
        return summary

    def _scaled_neural_test(self, bundle):
        scaler_path = self.scaler_root / f"{bundle.dataset_name}_scaler.pkl"
        if not scaler_path.exists():
            return bundle.X_test.astype("float64")
        scaler = joblib.load(scaler_path)
        X = bundle.X_test.astype("float64", copy=True)
        features = X.shape[-1]
        start = 1 if "Regime_ID" in bundle.train.columns else 0
        flat = X.reshape(-1, features)
        flat[:, start:] = scaler.transform(flat[:, start:])
        return flat.reshape(X.shape)

    def _model_input(self, model, bundle, X):
        try:
            multiple_inputs = len(model.inputs) == 2
        except (AttributeError, TypeError, ValueError):
            multiple_inputs = False
        if multiple_inputs:
            regime = bundle.test.sort_values(["Engine_ID", "Cycle"]).groupby("Engine_ID")["Regime_ID"].last().to_numpy()
            onehot = np.eye(REGIME_ONEHOT_SIZE, dtype="float32")[regime.astype(int)]
            return [X, onehot]
        return X

    @staticmethod
    def _predict(model, model_input, model_name):
        if model_name == "xgboost":
            return model.predict(model_input.reshape(model_input.shape[0], -1)).reshape(-1)
        return model.predict(model_input, verbose=0).reshape(-1)

    def _shap_values(self, model, model_name, X, model_input, feature_names):
        import shap

        n = min(100, len(X))
        background = X[:min(20, n)]
        sample = X[:n]
        if model_name == "xgboost":
            flat_background = background.reshape(len(background), -1)
            flat_sample = sample.reshape(len(sample), -1)
            try:
                explainer = shap.TreeExplainer(model)
                values = explainer.shap_values(flat_sample)
                # Do not convert explainer.expected_value: older SHAP builds
                # can return it as a string-formatted array.  The mean
                # background prediction is an equivalent numeric base value.
                expected = float(np.mean(model.predict(flat_background)))
                return (
                    np.asarray(values),
                    flat_sample,
                    expected,
                    feature_names * self.window_size,
                )
            except Exception as tree_error:
                # Compatibility fallback.  Some XGBoost/SHAP combinations
                # fail inside TreeExplainer because the serialized XGBoost
                # base_score is represented as ``"[7.53]"``.  Explain the
                # same model at sensor level instead of exposing SHAP to that
                # incompatible internal parameter.  Each sensor value is
                # repeated over the window, and the resulting SHAP values are
                # already in the same sensor-level form used by the report.
                sensor_background = background.mean(axis=1)
                sensor_sample = sample.mean(axis=1)

                def predict_from_sensor_values(sensor_values):
                    sensor_values = np.asarray(sensor_values, dtype="float32")
                    expanded = np.repeat(
                        sensor_values[:, None, :], self.window_size, axis=1
                    )
                    flat_expanded = expanded.reshape(len(expanded), -1)
                    return np.asarray(model.predict(flat_expanded)).reshape(-1)

                try:
                    fallback_explainer = shap.Explainer(
                        predict_from_sensor_values,
                        sensor_background,
                        algorithm="permutation",
                    )
                    fallback_result = fallback_explainer(
                        sensor_sample,
                        max_evals=2 * len(feature_names) + 1,
                    )
                    fallback_values = np.asarray(fallback_result.values)
                    fallback_base = getattr(fallback_result, "base_values", None)
                    expected = (
                        float(np.mean(fallback_base))
                        if fallback_base is not None
                        else float(np.mean(predict_from_sensor_values(sensor_background)))
                    )
                    print(
                        "SHAP TreeExplainer unavailable for XGBoost; "
                        "using sensor-level PermutationExplainer fallback: "
                        f"{tree_error}"
                    )
                    return fallback_values, sensor_sample, expected, feature_names
                except Exception as fallback_error:
                    raise RuntimeError(
                        "Both XGBoost TreeExplainer and sensor-level "
                        f"PermutationExplainer failed. TreeExplainer: {tree_error}; "
                        f"PermutationExplainer: {fallback_error}"
                    ) from fallback_error

        try:
            if isinstance(model_input, list):
                background_input = [array[:len(background)] for array in model_input]
                sample_input = [array[:n] for array in model_input]
            else:
                background_input = background
                sample_input = sample
            # A Sequential model may not expose the singular `model.input`
            # property even after fitting.  Calling it once materializes the
            # symbolic input/output tensors required by GradientExplainer.
            probe = ([array[:1] for array in sample_input]
                     if isinstance(sample_input, list) else sample_input[:1])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*structure of `inputs`.*", category=UserWarning)
                model(probe, training=False)
                explainer = shap.GradientExplainer(model, background_input)
                values = explainer.shap_values(sample_input)
            if isinstance(values, list):
                values = values[0]
            values = np.asarray(values)
            if values.ndim == 4:
                values = values[..., 0]
            expected = float(np.mean(self._predict(model, background_input, "neural")))
            return values, sample, expected, feature_names
        except Exception as error:
            raise RuntimeError(f"SHAP neural explainer failed: {error}") from error

    def _aggregate_sensor_shap(self, values, data, feature_names):
        # The XGBoost compatibility fallback explains one aggregated value
        # per sensor, so both SHAP values and data are already 2-D:
        # (samples, sensors).
        if values.ndim == 2 and data.ndim == 2:
            shap_agg = values
            data_agg = data
            names = feature_names
        elif values.ndim == 3:
            shap_agg = values.sum(axis=1)
            data_agg = data.mean(axis=1)
            names = feature_names
        else:
            time_steps = self.window_size
            n_features = len(feature_names)
            shap_agg = values.reshape(len(values), time_steps, n_features).sum(axis=1)
            data_agg = data.reshape(len(data), time_steps, n_features).mean(axis=1)
            names = feature_names
        return shap_agg, data_agg, names

    def generate_explainability(self, bundle, model_name, model, predictions):
        dataset_dir = self._dataset_dir(bundle.dataset_name)
        explain_dir = dataset_dir / "explainability" / model_name
        explain_dir.mkdir(parents=True, exist_ok=True)
        feature_names = self._feature_columns(bundle)
        X = bundle.X_test if model_name == "xgboost" else self._scaled_neural_test(bundle)
        model_input = X if model_name == "xgboost" else self._model_input(model, bundle, X)
        values, sample_data, expected, shap_names = self._shap_values(
            model, model_name, X, model_input, feature_names
        )
        agg_values, agg_data, agg_names = self._aggregate_sensor_shap(values, sample_data, feature_names)
        importance = np.mean(np.abs(agg_values), axis=0)
        order = np.argsort(importance)[::-1]

        # 10 and 11. SHAP summary and beeswarm.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            shap.summary_plot(agg_values, agg_data, feature_names=agg_names, show=False, max_display=20)
        self._savefig(explain_dir / "shap_summary.png")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            shap.summary_plot(agg_values, agg_data, feature_names=agg_names, plot_type="dot", show=False, max_display=20)
        self._savefig(explain_dir / "shap_beeswarm.png")

        # 12. SHAP feature importance.
        top = order[:20][::-1]
        plt.figure(figsize=(9, 7))
        plt.barh(np.array(agg_names)[top], importance[top])
        plt.xlabel("Mean absolute SHAP contribution")
        plt.title(f"{bundle.dataset_name}: SHAP Feature Importance")
        self._savefig(explain_dir / "shap_feature_importance.png")

        # 13. Local waterfall explanation.
        explanation = shap.Explanation(
            values=agg_values[0], base_values=expected, data=agg_data[0], feature_names=agg_names
        )
        shap.plots.waterfall(explanation, max_display=15, show=False)
        self._savefig(explain_dir / "shap_waterfall_sample_0.png")

        # 14. Sensor 11 dependence, if present.
        if "Sensor_11" in agg_names:
            index = agg_names.index("Sensor_11")
            plt.figure(figsize=(8, 6))
            plt.scatter(agg_data[:, index], agg_values[:, index], alpha=0.7)
            plt.xlabel("Sensor_11 aggregated value")
            plt.ylabel("SHAP contribution to predicted RUL")
            plt.title(f"{bundle.dataset_name}: Sensor_11 SHAP Dependence")
            self._savefig(explain_dir / "shap_dependence_sensor_11.png")

        pd.DataFrame({"Feature": agg_names, "MeanAbsSHAP": importance}).sort_values(
            "MeanAbsSHAP", ascending=False
        ).to_csv(explain_dir / "shap_feature_importance.csv", index=False)
        pd.DataFrame({"Feature": agg_names, "SHAP_Value": agg_values[0], "Feature_Value": agg_data[0]}).sort_values(
            "SHAP_Value", key=np.abs, ascending=False
        ).to_csv(explain_dir / "local_explanation_sample_0.csv", index=False)
        local_rows = []
        for sample_index in range(len(agg_values)):
            prediction = float(predictions[sample_index]) if sample_index < len(predictions) else np.nan
            actual = float(bundle.y_test[sample_index]) if sample_index < len(bundle.y_test) else np.nan
            for feature, value, shap_value in zip(agg_names, agg_data[sample_index], agg_values[sample_index]):
                local_rows.append({
                    "Sample": sample_index,
                    "Feature": feature,
                    "Feature_Value": value,
                    "SHAP_Value": shap_value,
                    "Influence": "increases predicted RUL" if shap_value > 0 else "decreases predicted RUL",
                    "Prediction": prediction,
                    "Actual_RUL": actual,
                })
        pd.DataFrame(local_rows).to_csv(explain_dir / "local_sensor_influences_all_samples.csv", index=False)

        # PDP and ICE for the most influential sensor, using all timesteps.
        top_sensor = agg_names[order[0]]
        self._plot_pdp_ice(model, model_name, bundle, X, model_input, feature_names, top_sensor, explain_dir)
        if "Sensor_11" in feature_names and top_sensor != "Sensor_11":
            self._plot_pdp_ice(model, model_name, bundle, X, model_input, feature_names, "Sensor_11", explain_dir)

    def _plot_pdp_ice(self, model, model_name, bundle, X, model_input, feature_names, sensor, output_dir):
        if sensor not in feature_names:
            return
        feature_index = feature_names.index(sensor)
        grid = np.linspace(np.percentile(X[..., feature_index], 5), np.percentile(X[..., feature_index], 95), 20) if X.ndim == 3 else np.linspace(np.percentile(X[:, feature_index], 5), np.percentile(X[:, feature_index], 95), 20)
        curves = []
        for value in grid:
            changed = X.copy()
            if changed.ndim == 3:
                changed[:, :, feature_index] = value
            else:
                changed[:, feature_index] = value
            inp = changed if model_name == "xgboost" else self._model_input(model, bundle, changed)
            curves.append(self._predict(model, inp, model_name))
        curves = np.asarray(curves).T
        plt.figure(figsize=(9, 6))
        for curve in curves[:min(30, len(curves))]:
            plt.plot(grid, curve, color="tab:blue", alpha=0.15)
        plt.plot(grid, curves.mean(axis=0), color="tab:red", linewidth=2, label="PDP mean")
        plt.xlabel(sensor)
        plt.ylabel("Predicted RUL")
        plt.title(f"{bundle.dataset_name}: PDP and ICE for {sensor}")
        plt.legend()
        plt.grid(alpha=0.25)
        self._savefig(output_dir / f"pdp_ice_{sensor.lower()}.png")

    def generate_model_plots(self, bundle, model_name, predictions, history):
        dataset_dir = self._dataset_dir(bundle.dataset_name)
        model_dir = dataset_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        y_true = bundle.y_test
        residuals = y_true - predictions
        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, predictions, alpha=0.7)
        low, high = min(y_true.min(), predictions.min()), max(y_true.max(), predictions.max())
        plt.plot([low, high], [low, high], "r--")
        plt.xlabel("Actual RUL"); plt.ylabel("Predicted RUL"); plt.title(f"{bundle.dataset_name} {model_name}: Prediction vs Actual")
        self._savefig(model_dir / "prediction_vs_actual.png")
        plt.figure(figsize=(7, 6)); plt.scatter(predictions, residuals, alpha=0.7); plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Predicted RUL"); plt.ylabel("Residual (actual - predicted)"); plt.title(f"{bundle.dataset_name} {model_name}: Residual Plot")
        self._savefig(model_dir / "residual_plot.png")
        plt.figure(figsize=(7, 5)); sns.histplot(residuals, bins=30, kde=True); plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.title(f"{bundle.dataset_name} {model_name}: Error Distribution")
        self._savefig(model_dir / "error_distribution.png")
        if history and "loss" in history and "val_loss" in history:
            plt.figure(figsize=(8, 5)); plt.plot(history["loss"], label="Training Huber loss"); plt.plot(history["val_loss"], label="Validation Huber loss"); plt.xlabel("Epoch"); plt.ylabel("Huber loss"); plt.title(f"{bundle.dataset_name} {model_name}: Training vs Validation Loss"); plt.legend(); plt.grid(alpha=0.25)
            self._savefig(model_dir / "training_vs_validation_loss.png")

    def generate_comparison(self, records):
        if not records:
            return
        metrics = pd.DataFrame(records)
        metrics.to_csv(self.root / "tables" / "all_model_metrics.csv", index=False)
        for dataset, group in metrics.groupby("Dataset"):
            long = group.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"], var_name="Metric", value_name="Value")
            plt.figure(figsize=(10, 6)); sns.barplot(data=long, x="Metric", y="Value", hue="Model"); plt.title(f"{dataset}: Model Metric Comparison"); plt.grid(axis="y", alpha=0.25); self._savefig(self._dataset_dir(dataset) / "models" / "metric_comparison.png")
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, metric in zip(axes, ("RMSE", "MAE", "R2")):
            sns.barplot(data=metrics, x="Dataset", y=metric, hue="Model", ax=ax)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Dataset-wise Grouped Model Comparison")
        self._savefig(self.root / "models" / "dataset_wise_grouped_comparison.png")

    @staticmethod
    def _validation_metric(model_info, metric):
        validation = model_info.get("validation_metrics", {})
        if metric in validation:
            return float(validation[metric])
        if metric == "RMSE" and "MSE" in validation:
            return float(np.sqrt(max(validation["MSE"], 0.0)))
        return np.nan

    def generate_selection_table(self, dataset, info):
        """Save the metric comparison used before the explicit SHAP choice."""
        rows = []
        for model_name, model_info in info["models"].items():
            test = model_info["metrics"].iloc[0]
            rows.append({
                "Dataset": dataset,
                "Model": model_name,
                "Validation_RMSE": self._validation_metric(model_info, "RMSE"),
                "Validation_MAE": self._validation_metric(model_info, "MAE"),
                "Validation_R2": self._validation_metric(model_info, "R2"),
                "Test_RMSE": float(test["RMSE"]),
                "Test_MAE": float(test["MAE"]),
                "Test_R2": float(test["R2"]),
            })
        table = pd.DataFrame(rows)
        for metric in ("RMSE", "MAE"):
            values = table[f"Validation_{metric}"].fillna(table[f"Test_{metric}"])
            table[f"{metric}_Rank"] = values.rank(method="min", ascending=True).astype(int)
        r2_values = table["Validation_R2"].fillna(table["Test_R2"])
        table["R2_Rank"] = r2_values.rank(method="min", ascending=False).astype(int)
        table["Average_Rank"] = table[["RMSE_Rank", "MAE_Rank", "R2_Rank"]].mean(axis=1)
        selected = SHAP_MODEL_BY_DATASET.get(dataset)
        table["Selected_For_SHAP"] = table["Model"].eq(selected)
        table["Selection_Basis"] = "RMSE + MAE + R2 comparison; configured research choice"
        table.to_csv(
            self._dataset_dir(dataset) / "tables" / "model_selection_comparison.csv",
            index=False,
        )

    def generate(self, results):
        records = []
        summaries = []
        for dataset, info in results.items():
            bundle = info["bundle"]
            summaries.append(self.generate_eda(bundle))
            for model_name, model_info in info["models"].items():
                self.generate_model_plots(bundle, model_name, model_info["predictions"], model_info.get("history"))
                metric = model_info["metrics"].iloc[0].to_dict()
                records.append({"Dataset": dataset, "Model": model_name, **metric})
            self.generate_selection_table(dataset, info)
            selected_model = SHAP_MODEL_BY_DATASET.get(dataset, info.get("best_model"))
            if selected_model in info["models"]:
                info["best_model"] = selected_model
                best = info["models"][selected_model]
                error_path = self._dataset_dir(dataset) / "explainability" / "shap_error.txt"
                try:
                    self.generate_explainability(bundle, selected_model, best["model"], best["predictions"])
                    # Remove an error marker left by an earlier run after a
                    # later run completes successfully.
                    if error_path.exists():
                        error_path.unlink()
                except Exception as error:
                    error_path.write_text(str(error), encoding="utf-8")
                    print(f"WARNING: SHAP generation failed for {dataset}: {error}")
        self.generate_comparison(records)
        if summaries:
            pd.concat(summaries, ignore_index=True).to_csv(
                self.root / "tables" / "data_summary_all_datasets.csv", index=False
            )
