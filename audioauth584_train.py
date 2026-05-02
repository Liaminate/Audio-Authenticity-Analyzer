import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from audioauth584_features import extract_features, FEATURE_NAMES

# path
DATA_ROOT  = "data"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "audioauth584_model.pkl")
NAMES_PATH = os.path.join(MODEL_DIR, "audioauth584_feature_names.pkl")

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


# dataset folders and extract features from every audio file

def load_dataset(data_root: str):
    # Labels: 0 = real,  1 = fake/AI-generated
    X, y = [], []
    label_map = {"real": 0, "fake": 1}

    for class_name, label in label_map.items():
        class_dir = os.path.join(data_root, class_name)

        if not os.path.isdir(class_dir):
            print(f"  WARNING: folder not found — {class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]

        print(f"  Loading {len(files)} files from '{class_name}/'...")

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                features = extract_features(fpath)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"    Skipping {fname}: {e}")

    return np.array(X, dtype=np.float64), np.array(y, dtype=int)


# trian random forest

def train(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing  samples : {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n========== EVALUATION RESULTS ==========")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives  (real -> real) : {cm[0][0]}")
    print(f"  False Positives (real -> fake) : {cm[0][1]}")
    print(f"  False Negatives (fake -> real) : {cm[1][0]}")
    print(f"  True Positives  (fake -> fake) : {cm[1][1]}")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score : {auc:.4f}  (1.0 = perfect, 0.5 = random)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"5-Fold CV AUC : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("\nFeature Importances (most to least influential):")
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda t: t[1], reverse=True)
    for name, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"  {name:<25} {bar}  ({imp:.4f})")

    return model


# save model

def save_model(model: RandomForestClassifier):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURE_NAMES, NAMES_PATH)
    print(f"\nModel saved  -> {MODEL_PATH}")
    print(f"Names saved  -> {NAMES_PATH}")


# entry
if __name__ == "__main__":
    print("=== audioauth584 — Model Training ===\n")
    print(f"Dataset root : {DATA_ROOT}")

    print("\nExtracting features from dataset...")
    X, y = load_dataset(DATA_ROOT)

    if len(X) == 0:
        print("\nERROR: No audio files were loaded.")
        print("Make sure data/real/ and data/fake/ exist and contain .wav or .mp3 files.")
        raise SystemExit(1)

    real_count = int(np.sum(y == 0))
    fake_count = int(np.sum(y == 1))
    print(f"\nDataset summary: {real_count} real, {fake_count} fake ({len(X)} total)")

    if real_count == 0 or fake_count == 0:
        print("ERROR: Need at least one file in each class (real/ and fake/).")
        raise SystemExit(1)

    model = train(X, y)
    save_model(model)

    print("\nDone! Launch the GUI with: python audioauth584_main.py")
