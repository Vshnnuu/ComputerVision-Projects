import joblib
from src.config import DATA_PATH, OUTPUT_DIR
from src.data_prep import load_and_prepare_data
from src.train import train_model
from src.utils import ensure_output_dir


def main():
    ensure_output_dir(OUTPUT_DIR)

    print("Loading and preparing data...")
    prepared = load_and_prepare_data(DATA_PATH)

    print("Saving scaler...")
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    joblib.dump(prepared.scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print("Starting training...")
    train_model(
        prepared.X_train,
        prepared.X_test,
        prepared.y_train,
        prepared.y_test,
    )

    print("Done.")


if __name__ == "__main__":
    main()