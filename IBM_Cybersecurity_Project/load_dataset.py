import pandas as pd

print("=" * 60)
print("AI CYBERSECURITY NETWORK THREAT & INTRUSION PROFILER")
print("=" * 60)

# Load Dataset
train_data = pd.read_csv(
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

train_data.columns = columns

print("\n✅ Dataset Loaded Successfully!")

print("\nDataset Shape:")
print(train_data.shape)

print("\nColumn Names:\n")
print(train_data.columns)

print("\nFirst Five Rows:\n")
print(train_data.head())

print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)

# Dataset Information
print("\nDataset Info:\n")
print(train_data.info())

# Missing Values
print("\nMissing Values:\n")
print(train_data.isnull().sum())

# Attack Types
print("\nUnique Attack Types:\n")
print(train_data["attack"].unique())

# Number of Attack Categories
print("\nAttack Counts:\n")
print(train_data["attack"].value_counts())