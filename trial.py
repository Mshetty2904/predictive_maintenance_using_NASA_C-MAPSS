from config import MODEL_PATH
from pathlib import Path
print(MODEL_PATH)
print(type(MODEL_PATH))
print("=" * 60)
def __init__(self, model_path, scaler_path):

        self.model_path = Path(model_path)
        self.scaler_path = scaler_path

        self.model_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        model_file = (
                    self.model_path
                    / f"{bundle.dataset_name}_cnn_lstm.keras"
                    print("=" * 60)
                    print("Dataset:", bundle.dataset_name),
                    print("Model file:", repr(model_file)),
                    print("As string:", str(model_file)),
                    print("Parent exists:", model_file.parent.exists()),
                    print("Parent is dir:", model_file.parent.is_dir()),
                    print("=" * 60),
                )
        
       