
from pathlib import Path

import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# 1. KONFIGURASI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "dataset_5_class_animal_preprocessing"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = BASE_DIR / "model"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42
LEARNING_RATE = 0.001

MLRUNS_DIR = PROJECT_DIR / "mlruns"
mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())


# 2. VALIDASI DATASET

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {folder}"
        )

tf.keras.utils.set_random_seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes != 5:
    raise ValueError(
            f"Ditemukan {num_classes} kelas, seharusnya 5."
    )

if val_ds.class_names != class_names:
    raise ValueError("Urutan kelas train dan validation berbeda.")

if test_ds.class_names != class_names:
    raise ValueError("Urutan kelas train dan test berbeda.")

AUTOTUNE = tf.data.AUTOTUNE

def normalize_dataset(dataset):
    return dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

train_ds = normalize_dataset(train_ds)
val_ds = normalize_dataset(val_ds)
test_ds = normalize_dataset(test_ds)


# 3. MEMBANGUN MODEL CNN

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# 4. MLFLOW AUTOLOG DAN MELATIH MODEL

mlflow.tensorflow.autolog(
    log_models=True,
    log_every_epoch=True
)

with mlflow.start_run(run_name="Workflow_CI") as run:
    
    model = build_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        ]
    )


    # 5. EVALUASI PADA TEST SET

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    mlflow.log_metrics({
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
    })

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    mlflow.log_text(
        report,
        artifact_file="classification_report.txt"
    )


    # 6. CONFUSION MATRIX

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    
    fig, ax = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    mlflow.log_figure(
        fig,
        "confusion_matrix.png"
    )

    plt.close(fig)


    # 7. GRAFIK TRAINING 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()

    mlflow.log_figure(fig, "training_curves.png")

    plt.close(fig)

    mlflow.log_dict({"class_names": class_names}, "class_names.json")


    # 8. MEMBUAT SIGNATURE MODEL

    sample_images, _ = next(iter(test_ds.take(1)))

    sample_predictions = model.predict(sample_images, verbose=0)

    signature = infer_signature(sample_images.numpy(), sample_predictions)


    # 9. SIMPAN MODEL SEBAGAI MLFLOW MODEL

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in MODEL_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()
        elif old_file.is_dir():
            import shutil
            shutil.rmtree(old_file)


    mlflow.tensorflow.save_model(
        model=model,
        path=str(MODEL_DIR),
        signature=signature
    )


    # 10. RINGKASAN HASIL

    print("=" * 55)
    print("HASIL EKSPERIMEN CNN BASIC")
    print("Kelas:", class_names)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("MLflow Run ID:", run.info.run_id)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Model directory:", MODEL_DIR)
    print("Model signature:", signature)
    print("=" * 55)

print("\nEksperimen Basic selesai.")
