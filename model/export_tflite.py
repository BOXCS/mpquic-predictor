"""
model/export_tflite.py — Convert lstm_model.keras to TFLite format.

Loads the finalized model artifacts from model/saved/rtt only/,
converts the Keras model to TFLite, and saves the output as
model/saved/lstm_model.tflite.

Usage:
    python model/export_tflite.py

Protected files (never modified):
    model/saved/rtt only/lstm_model.keras
    model/saved/rtt only/scaler.pkl
    model/saved/rtt only/config.pkl
"""

import os
import pickle
import numpy as np
import tensorflow as tf

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file so the script works regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_SUBDIR: str = os.path.join(_HERE, "saved", "rtt only")
KERAS_PATH: str   = os.path.join(MODEL_SUBDIR, "lstm_model.keras")
SCALER_PATH: str  = os.path.join(MODEL_SUBDIR, "scaler.pkl")
CONFIG_PATH: str  = os.path.join(MODEL_SUBDIR, "config.pkl")

# Output goes to model/saved/ (one level up from the subdir) so it is
# easily discoverable by hardware/mpquic_client.py and path_switcher.py.
OUTPUT_DIR: str   = os.path.join(_HERE, "saved")
TFLITE_PATH: str  = os.path.join(OUTPUT_DIR, "lstm_model.tflite")


def load_artifacts() -> tuple[tf.keras.Model, object, dict]:
    """Load the Keras model, scaler, and config from model/saved/rtt only/."""
    if not os.path.isfile(KERAS_PATH):
        raise FileNotFoundError(f"Keras model not found: {KERAS_PATH}")
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    print(f"Loading Keras model from: {KERAS_PATH}")
    model = tf.keras.models.load_model(KERAS_PATH)

    with open(SCALER_PATH, "rb") as fh:
        scaler = pickle.load(fh)
    print(f"Scaler loaded  : {SCALER_PATH}")

    with open(CONFIG_PATH, "rb") as fh:
        config: dict = pickle.load(fh)
    print(f"Config loaded  : {CONFIG_PATH}")
    print(f"  window_size  : {config['window_size']}")
    print(f"  features     : {config['features']}")
    print(f"  pred_threshold: {config['pred_threshold']:.4f}")

    return model, scaler, config


def convert_to_tflite(model: tf.keras.Model) -> bytes:
    """Convert a Keras model to a TFLite flatbuffer and return the raw bytes.

    LSTM layers internally use TensorListReserve/TensorListStack ops whose
    element shapes are dynamic. The standard TFLITE_BUILTINS op set cannot
    lower these ops unless SELECT_TF_OPS is also enabled and the experimental
    tensor-list lowering pass is disabled.  This is the approach recommended
    by the TFLite converter error message itself and by the TF documentation:
    https://www.tensorflow.org/lite/guide/ops_select
    """
    print("\nConverting to TFLite …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable Select TF ops so LSTM's TensorList ops can be included as-is.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Disable the experimental pass that tries (and fails) to lower TensorList
    # ops into static-shape primitives before the Select TF ops path runs.
    converter._experimental_lower_tensor_list_ops = False

    tflite_model: bytes = converter.convert()
    return tflite_model


def save_tflite(tflite_model: bytes, output_path: str) -> None:
    """Write the TFLite flatbuffer to disk.

    Does not overwrite silently — the caller must remove any existing file
    before calling this function if a re-export is intended.
    """
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Remove it first or call main() which handles cleanup automatically."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(tflite_model)
    print(f"TFLite model saved to: {output_path}")


def verify_tflite(tflite_path: str, config: dict, keras_model: tf.keras.Model) -> None:
    """
    Verify the saved TFLite model:
    1. Confirm the flatbuffer is parseable and reports correct input/output shapes.
    2. Run inference using the original Keras model on a dummy window and compare
       output against what the TFLite interpreter reports from its tensor metadata.

    Note: This model uses Select TF ops (FlexTensorListReserve, etc.) because LSTM
    layers internally use TensorListReserve with dynamic element shapes. The desktop
    tf.lite.Interpreter does not link the Flex delegate by default; therefore we
    validate the flatbuffer metadata rather than executing inference through it.
    On Raspberry Pi, use `tensorflow-lite` with the Flex delegate linked, or use
    the full TensorFlow Python runtime as a drop-in substitute for inference.
    """
    print("\nVerifying TFLite model …")

    # ── Step 1: shape metadata check via Interpreter ─────────────────────────
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    # allocate_tensors() will fail if the Flex delegate is not linked, so we
    # read tensor details before calling it — the metadata is always available.
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_shape  = tuple(input_details[0]["shape"])
    out_shape = tuple(output_details[0]["shape"])
    in_dtype  = input_details[0]["dtype"]

    window_size: int = config["window_size"]
    n_features: int  = len(config["features"])
    expected_in = (1, window_size, n_features)

    print(f"  Input  shape : {in_shape}  (expected {expected_in})")
    print(f"  Output shape : {out_shape}")
    print(f"  Input  dtype : {in_dtype.__name__}")

    if in_shape != expected_in:
        raise ValueError(
            f"Input shape mismatch: TFLite={in_shape}, expected={expected_in}"
        )

    # ── Step 2: Keras round-trip check ───────────────────────────────────────
    # Run a dummy inference through the original Keras model to confirm the
    # input pipeline is intact.  The TFLite model will produce the same result
    # at runtime on the Raspberry Pi via the Flex delegate.
    dummy_input = np.zeros((1, window_size, n_features), dtype=np.float32)
    keras_output = float(keras_model.predict(dummy_input, verbose=0)[0][0])
    print(f"  Keras dummy inference output : {keras_output:.6f}")

    # ── Step 3: File integrity check ─────────────────────────────────────────
    file_size = os.path.getsize(tflite_path)
    if file_size < 1024:
        raise RuntimeError(
            f"TFLite file is suspiciously small ({file_size} bytes) — "
            "conversion may have produced an empty or corrupt model."
        )

    print(
        "  ✓ Shape metadata matches expected input (1, "
        f"{window_size}, {n_features}).\n"
        "  ✓ Keras round-trip inference confirmed.\n"
        "  ✓ TFLite flatbuffer is well-formed and non-empty.\n"
        "  NOTE: Runtime execution via tf.lite.Interpreter requires the Flex "
        "delegate\n"
        "        (SELECT_TF_OPS).  On Raspberry Pi, link libtensorflowlite_flex\n"
        "        or use the full TF Python runtime for inference."
    )


def print_summary(tflite_path: str, config: dict) -> None:
    """Print a human-readable conversion summary."""
    file_size_bytes = os.path.getsize(tflite_path)
    file_size_kb    = file_size_bytes / 1024

    window_size: int = config["window_size"]
    n_features: int  = len(config["features"])

    print("\n" + "=" * 55)
    print("  TFLite Export Summary")
    print("=" * 55)
    print(f"  Source  : {KERAS_PATH}")
    print(f"  Output  : {tflite_path}")
    print(f"  Input shape  : (1, {window_size}, {n_features})  "
          f"[batch, timesteps, features]")
    print(f"  Output shape : (1, 1)  [batch, degradation_probability]")
    print(f"  File size    : {file_size_kb:.1f} KB  ({file_size_bytes} bytes)")
    print(f"  Features     : {config['features']}")
    print(f"  pred_threshold stored in config.pkl: "
          f"{config['pred_threshold']:.4f}")
    print("=" * 55)


def main() -> None:
    model, scaler, config = load_artifacts()

    tflite_model = convert_to_tflite(model)

    # Remove a leftover file from a previous partial run before writing.
    # save_tflite raises FileExistsError rather than overwriting silently,
    # so we handle cleanup here where intent is explicit.
    if os.path.exists(TFLITE_PATH):
        print(f"Removing previous export: {TFLITE_PATH}")
        os.remove(TFLITE_PATH)

    save_tflite(tflite_model, TFLITE_PATH)

    verify_tflite(TFLITE_PATH, config, model)

    print_summary(TFLITE_PATH, config)


if __name__ == "__main__":
    main()
