import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")
    return df


def preprocess(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    # Use class weights instead of SMOTE to avoid synthetic data artifacts
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    return X_train, X_test, y_train, y_test, scaler, class_weight


def build_model(input_dim: int) -> keras.Model:
    reg = keras.regularizers.l2(1e-3)
    model = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(input_dim,),
                               kernel_regularizer=reg),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(curve="ROC", name="auc_roc"),
            keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )
    return model


def train_model(model: keras.Model, X_train, y_train, class_weight) -> keras.callbacks.History:
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=2048,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def plot_training_history(history: keras.callbacks.History):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()


def evaluate_model(model: keras.Model, X_test, y_test, threshold: float = 0.7):
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > threshold).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = average_precision_score(y_test, y_pred_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=150)
    plt.show()


def save_artifacts(model: keras.Model, scaler: StandardScaler):
    model.save("fraud_model.keras")
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved fraud_model.keras, fraud_model.pkl, and scaler.pkl")


def main():
    print("Loading data...")
    df = load_data("creditcard.csv")
    print(f"Dataset shape: {df.shape}")

    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test, scaler, class_weight = preprocess(df)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Class weights: {class_weight}")

    print("\nBuilding model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    print("\nTraining model...")
    history = train_model(model, X_train, y_train, class_weight)
    plot_training_history(history)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test, threshold=0.7)

    print("\nSaving artifacts...")
    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()



