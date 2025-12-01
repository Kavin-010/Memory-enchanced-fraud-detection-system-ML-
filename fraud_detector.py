"""
Fraud Detection using Autoencoder (Memory-Enhanced)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
LATENT_DIM, EPOCHS, BATCH_SIZE, TEST_BATCH = 16, 30, 64, 100


class FraudDetector:
    def __init__(self):
        self.scaler, self.model, self.encoder, self.memory, self.threshold = None, None, None, None, None

    # --------------------------------------------------------
    def build_model(self, dim):
        inp = keras.Input(shape=(dim,))
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dense(32, activation='relu')(x)
        z = layers.Dense(LATENT_DIM)(x)
        x = layers.Dense(32, activation='relu')(z)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(dim, activation='sigmoid')(x)
        self.model = keras.Model(inp, out)
        self.encoder = keras.Model(inp, z)
        self.model.compile(optimizer='sgd', loss='mse')

    # --------------------------------------------------------
    def train(self):
        print("[1/4] Loading data...")
        df = pd.read_csv("train.csv")
        data = df[df['Class'] == 0].drop(columns=['Class']) if 'Class' in df.columns else df
        print(f"      {len(data)} normal samples loaded")

        print("[2/4] Training autoencoder...")
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(data.values)
        self.build_model(X.shape[1])
        self.model.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0)

        print("[3/4] Building memory and setting threshold...")
        self.memory = self.encoder.predict(X, verbose=0)
        recon = self.model.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=1)
        self.threshold = np.mean(errors) + 3 * np.std(errors)

        print("[4/4] Saving models and memory...")
        self.model.save("model.keras")
        self.encoder.save("encoder.keras")
        joblib.dump(self.scaler, "scaler.joblib")
        np.save("memory.npy", self.memory)
        np.savetxt("threshold.txt", [self.threshold])
        print(f"✓ Done! Memory: {len(self.memory)}, Threshold: {self.threshold:.6f}")

    # --------------------------------------------------------
    def load(self):
        self.model = keras.models.load_model("model.keras")
        self.encoder = keras.models.load_model("encoder.keras")
        self.scaler = joblib.load("scaler.joblib")
        self.memory = np.load("memory.npy")
        self.threshold = float(np.loadtxt("threshold.txt"))
        print(f"Loaded: Memory={len(self.memory)}, Threshold={self.threshold:.6f}")

    # --------------------------------------------------------
    def add_unique(self, vecs):
        added, duplicates = 0, 0
        for v in vecs:
            is_duplicate = any(np.allclose(v, mem_vec, rtol=1e-9, atol=1e-9) for mem_vec in self.memory)
            if is_duplicate:
                duplicates += 1
            else:
                self.memory = np.vstack([self.memory, v.reshape(1, -1)])
                added += 1

        if added > 0:
            np.save("memory.npy", self.memory)
        return added, duplicates

    # --------------------------------------------------------
    def detect(self, X):
        latent = self.encoder.predict(X, verbose=0)
        preds, errors = [], []

        for i, x_latent in enumerate(latent):
            in_memory = any(np.allclose(x_latent, mem_vec, rtol=1e-9, atol=1e-9) for mem_vec in self.memory)

            recon = self.model.predict(X[i].reshape(1, -1), verbose=0)
            err = np.mean((X[i] - recon[0]) ** 2)
            errors.append(err)

            if in_memory:
                preds.append(0)
            else:
                preds.append(1 if err > self.threshold else 0)

        return np.array(preds), np.array(errors)

    # --------------------------------------------------------
    def plot_visualizations(self, y_true, preds, errors):
        accuracy = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Accuracy: {accuracy:.2%}', fontsize=16, fontweight='bold', y=1.02)

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        ax[0].set_title('Confusion Matrix')
        ax[0].set_xlabel('Predicted')
        ax[0].set_ylabel('Actual')

        # Safe ROC curve plot
        try:
            if len(np.unique(y_true)) < 2:
                raise ValueError("ROC requires both classes present.")
            fpr, tpr, _ = roc_curve(y_true, errors)
            roc_auc = auc(fpr, tpr)
            ax[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax[1].plot([0, 1], [0, 1], 'k--')
            ax[1].set_xlabel('False Positive Rate')
            ax[1].set_ylabel('True Positive Rate')
            ax[1].set_title('ROC Curve')
            ax[1].legend(loc='lower right')
        except Exception:
            ax[1].text(0.5, 0.5, 'ROC not available\n(single-class batch)',
                       ha='center', va='center', fontsize=12)
            ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()

        # Classification Report
        print("\n=== Classification Report ===")
        print(classification_report(y_true, preds, target_names=['Normal', 'Fraud']))
        print(f"✓ Accuracy: {accuracy:.2%}")

    # --------------------------------------------------------
    def process_batch(self):
        df = pd.read_csv("test.csv")
        if len(df) == 0:
            print("No test data left!")
            return

        size = min(TEST_BATCH, len(df))
        batch = df.iloc[:size]
        df.iloc[size:].to_csv("test.csv", index=False)

        has_labels = 'Class' in batch.columns
        if has_labels:
            y_true = batch['Class'].values
            X = self.scaler.transform(batch.drop(columns=['Class']).values)
        else:
            X = self.scaler.transform(batch.values)
            y_true = None

        preds, errors = self.detect(X)

        # Update memory with normal samples
        normal_mask = (preds == 0)
        added, duplicates = 0, 0
        if np.any(normal_mask):
            latent = self.encoder.predict(X[normal_mask], verbose=0)
            added, duplicates = self.add_unique(latent)

        print("\n" + "=" * 60)
        print(f"Processed batch size: {len(batch)}")
        if has_labels:
            print(f"  Actual frauds: {np.sum(y_true == 1)}")
        print(f"  Detected frauds: {np.sum(preds == 1)}")
        print(f"  Normal samples: {np.sum(preds == 0)}")
        print(f"Memory Update: Added={added}, Duplicates={duplicates}, Total={len(self.memory)}")

        if has_labels:
            self.plot_visualizations(y_true, preds, errors)
        else:
            print("No labels available — skipping accuracy/confusion matrix.")

        print(f"Remaining samples in test.csv: {len(df) - size}")
        print("=" * 60)


# --------------------------------------------------------
def main():
    detector = FraudDetector()

    if os.path.exists("model.keras"):
        print("Loading model...\n")
        detector.load()

        if os.path.exists("test.csv"):
            detector.process_batch()
        else:
            print("No test.csv found!")
    else:
        print("Training new model...\n")
        if os.path.exists("train.csv"):
            detector.train()
            print("\n✓ Run again to process test data")
        else:
            print("ERROR: train.csv not found!")


# --------------------------------------------------------
if __name__ == "__main__":
    main()
