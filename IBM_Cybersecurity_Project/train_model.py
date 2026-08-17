import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 60)
print("AI CYBERSECURITY NETWORK THREAT & INTRUSION PROFILER")
print("=" * 60)

# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv(
    "dataset/KDDTrain+.txt",
    sep="\t",
    header=None
)

# =========================================================
# 2. COLUMN NAMES
# =========================================================

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack", "difficulty"
]

data.columns = columns

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# =========================================================
# 3. CREATE SEPARATE ENCODERS
# =========================================================

protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()
attack_encoder = LabelEncoder()

# =========================================================
# 4. ENCODE CATEGORICAL FEATURES
# =========================================================

data["protocol_type"] = protocol_encoder.fit_transform(
    data["protocol_type"]
)

data["service"] = service_encoder.fit_transform(
    data["service"]
)

data["flag"] = flag_encoder.fit_transform(
    data["flag"]
)

data["attack"] = attack_encoder.fit_transform(
    data["attack"]
)

print("\nCategorical Encoding Completed!")

# =========================================================
# 5. FEATURES AND TARGET
# =========================================================

X = data.drop(
    ["attack", "difficulty"],
    axis=1
)

y = data["attack"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))

# =========================================================
# 7. RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest Model...")

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# =========================================================
# 8. MODEL PREDICTION
# =========================================================

predictions = model.predict(X_test)

# =========================================================
# 9. ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =========================================================
# 11. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

# =========================================================
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

# =========================================================
# 13. SAVE ENCODERS
# =========================================================

encoders = {
    "protocol": protocol_encoder,
    "service": service_encoder,
    "flag": flag_encoder,
    "attack": attack_encoder
}

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print("\nSaved Files:")
print("models/random_forest_model.pkl")
print("models/encoders.pkl")

print("\nProject AI Training Completed!")