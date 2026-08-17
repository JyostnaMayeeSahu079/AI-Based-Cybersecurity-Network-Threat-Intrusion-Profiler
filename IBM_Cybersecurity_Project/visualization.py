import pandas as pd
import matplotlib.pyplot as plt

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

# -----------------------------
# Graph 1 - Bar Chart
# -----------------------------
attack_counts = data["attack"].value_counts().head(10)

plt.figure(figsize=(10,6))
attack_counts.plot(kind="bar")
plt.title("Top 10 Attack Types")
plt.xlabel("Attack Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/attack_distribution.png")
plt.close()

# -----------------------------
# Graph 2 - Pie Chart
# -----------------------------
plt.figure(figsize=(8,8))
attack_counts.plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.ylabel("")
plt.title("Top 10 Attack Types Distribution")
plt.tight_layout()
plt.savefig("images/attack_pie_chart.png")
plt.close()

# -----------------------------
# Graph 3 - Histogram
# -----------------------------
plt.figure(figsize=(10,6))
data["src_bytes"].hist(bins=30)

plt.title("Source Bytes Distribution")
plt.xlabel("Source Bytes")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("images/histogram_src_bytes.png")
plt.close()

# -----------------------------
# Graph 4 - Correlation Heatmap
# -----------------------------

numeric_data = data.select_dtypes(include=["number"])

correlation = numeric_data.corr()

plt.figure(figsize=(12,10))
plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/correlation_heatmap.png")
plt.close()


print("✅ Bar Chart Saved")
print("✅ Pie Chart Saved")
print("✅ Histogram Saved")
print("✅ Correlation Heatmap Saved")