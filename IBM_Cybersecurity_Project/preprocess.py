import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Load Dataset
data = pd.read_csv(
    "dataset/KDDTrain+.txt",
    sep="\t",
    header=None
)

# Column Names
columns = [
    "duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell",
    "su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "difficulty"
]

data.columns = columns

# Encode categorical columns
encoder = LabelEncoder()

data["protocol_type"] = encoder.fit_transform(data["protocol_type"])
data["service"] = encoder.fit_transform(data["service"])
data["flag"] = encoder.fit_transform(data["flag"])
data["attack"] = encoder.fit_transform(data["attack"])

print("\n✅ Encoding Completed Successfully!")

print("\nFirst 5 Rows:\n")
print(data.head())

print("\nDataset Shape:")
print(data.shape)