import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Cybersecurity Threat Profiler",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

.main-title {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    color: #00d4ff;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 18px;
}

.section-title {
    color: #00d4ff;
    font-size: 23px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

.metric-card {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    min-height: 90px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
}

.metric-value {
    color: #00d4ff;
    font-size: 22px;
    font-weight: 700;
    margin-top: 5px;
}

.threat-high {
    background: #35151a;
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.threat-medium {
    background: #352b12;
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.threat-low {
    background: #10291e;
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.threat-title {
    font-size: 24px;
    font-weight: 700;
}

.small-note {
    color: #94a3b8;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# COLUMN NAMES
# =========================================================

columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
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


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(
        "models/random_forest_model.pkl"
    )

    encoders = joblib.load(
        "models/encoders.pkl"
    )

    model_loaded = True

except Exception as e:

    model_loaded = False
    model = None
    encoders = None

    st.error(
        f"Model loading error: {e}"
    )


# =========================================================
# LOAD TRAINING DATA
# =========================================================

try:

    train_data = pd.read_csv(
        "dataset/KDDTrain+.txt",
        sep="\t",
        header=None
    )

    train_data.columns = columns

except Exception as e:

    st.error(
        f"Training dataset error: {e}"
    )

    st.stop()


# =========================================================
# CALCULATE MODEL PERFORMANCE
# =========================================================

performance_ready = False

try:

    X = train_data.drop(
        columns=["attack", "difficulty"]
    ).copy()

    y = train_data["attack"].copy()

    X["protocol_type"] = (
        encoders["protocol"].transform(
            X["protocol_type"]
        )
    )

    X["service"] = (
        encoders["service"].transform(
            X["service"]
        )
    )

    X["flag"] = (
        encoders["flag"].transform(
            X["flag"]
        )
    )

    y = encoders["attack"].transform(y)

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    performance_ready = True

except Exception as e:

    st.warning(
        f"Performance calculation error: {e}"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ AI CYBERSECURITY NETWORK THREAT PROFILER'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Intelligence & Machine Learning Based Intrusion Detection System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TOP STATUS CARDS
# =========================================================

status1, status2, status3, status4 = st.columns(4)


with status1:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">System</div>
        <div class="metric-value">🟢 ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)


with status2:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">AI Algorithm</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)


with status3:

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Dataset</div>
        <div class="metric-value">NSL-KDD</div>
    </div>
    """, unsafe_allow_html=True)


with status4:

    if model_loaded:
        status_text = "🟢 READY"
    else:
        status_text = "🔴 ERROR"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Overview",
        "📈 Analytics",
        "🤖 AI Prediction",
        "ℹ️ About Project"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        '📊 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    if performance_ready:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with c2:
            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with c3:
            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with c4:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )


    st.write("")


    # Confusion matrix
    if performance_ready:

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### 🔲 Confusion Matrix"
            )

            cm = confusion_matrix(
                y,
                predictions
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            ax.imshow(
                cm,
                interpolation="nearest"
            )

            ax.set_title(
                "Model Confusion Matrix"
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            for i in range(cm.shape[0]):

                for j in range(cm.shape[1]):

                    ax.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        fontsize=8
                    )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        with right:

            st.markdown(
                "### 📋 Performance Summary"
            )

            summary_data = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score"
                    ],
                    "Score": [
                        f"{accuracy * 100:.2f}%",
                        f"{precision * 100:.2f}%",
                        f"{recall * 100:.2f}%",
                        f"{f1 * 100:.2f}%"
                    ]
                }
            )

            st.dataframe(
                summary_data,
                hide_index=True,
                use_container_width=True
            )

            st.info(
                "The Random Forest model learns "
                "patterns from network traffic and "
                "classifies potential intrusion types."
            )


# =========================================================
# TAB 2 — ANALYTICS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        '📈 Security Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    attack_counts = (
        train_data["attack"]
        .value_counts()
        .head(10)
    )

    chart1, chart2 = st.columns(2)


    # Bar chart
    with chart1:

        st.markdown(
            "### 🚨 Top Attack Types"
        )

        fig1, ax1 = plt.subplots(
            figsize=(6, 4)
        )

        attack_counts.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_xlabel(
            "Attack Type"
        )

        ax1.set_ylabel(
            "Records"
        )

        ax1.tick_params(
            axis="x",
            rotation=45,
            labelsize=8
        )

        plt.tight_layout()

        st.pyplot(
            fig1,
            use_container_width=True
        )

        plt.close(fig1)


    # Pie chart
    with chart2:

        st.markdown(
            "### 🥧 Attack Distribution"
        )

        fig2, ax2 = plt.subplots(
            figsize=(6, 4)
        )

        attack_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax2,
            textprops={"fontsize": 8}
        )

        ax2.set_ylabel("")

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )

        plt.close(fig2)


    st.markdown(
        "### 📋 Attack Statistics"
    )

    stats_df = pd.DataFrame(
        {
            "Attack Type": attack_counts.index,
            "Records": attack_counts.values
        }
    )

    st.dataframe(
        stats_df,
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# TAB 3 — AI PREDICTION
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        '🤖 AI Network Threat Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select a network record and let the trained "
        "Random Forest model analyze it."
    )


    # Load test dataset
    try:

        test_data = pd.read_csv(
            "dataset/KDDTest+.txt",
            sep="\t",
            header=None
        )

        if test_data.shape[1] == 42:

            test_data.columns = columns[:-1]

        elif test_data.shape[1] == 43:

            test_data.columns = columns

        else:

            st.error(
                f"Unexpected dataset format: "
                f"{test_data.shape[1]} columns"
            )

            st.stop()


        # Record selection
        record_number = st.number_input(
            "Select Network Record",
            min_value=0,
            max_value=len(test_data) - 1,
            value=0,
            step=1
        )


        selected_record = test_data.iloc[
            [int(record_number)]
        ]


        # Compact record display
        with st.expander(
            "📡 View Selected Network Record"
        ):

            st.dataframe(
                selected_record,
                use_container_width=True
            )


        analyze = st.button(
            "🔍 ANALYZE NETWORK TRAFFIC",
            use_container_width=True
        )


        if analyze:

            try:

                prediction_data = selected_record.drop(
                    columns=[
                        "attack",
                        "difficulty"
                    ],
                    errors="ignore"
                ).copy()


                prediction_data[
                    "protocol_type"
                ] = encoders["protocol"].transform(
                    prediction_data["protocol_type"]
                )


                prediction_data[
                    "service"
                ] = encoders["service"].transform(
                    prediction_data["service"]
                )


                prediction_data[
                    "flag"
                ] = encoders["flag"].transform(
                    prediction_data["flag"]
                )


                prediction = model.predict(
                    prediction_data
                )[0]


                predicted_attack = (
                    encoders["attack"]
                    .inverse_transform(
                        [prediction]
                    )[0]
                )


                actual_attack = selected_record[
                    "attack"
                ].iloc[0]


                confidence = None

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = model.predict_proba(
                        prediction_data
                    )[0]

                    confidence = (
                        max(probabilities) * 100
                    )


                # Result cards
                r1, r2, r3 = st.columns(3)


                with r1:

                    st.metric(
                        "Predicted Attack",
                        str(predicted_attack)
                    )


                with r2:

                    if (
                        str(actual_attack).lower()
                        == "attack"
                    ):

                        actual_display = "🔴 ATTACK"

                    else:

                        actual_display = "🟢 NORMAL"


                    st.metric(
                        "Actual Label",
                        actual_display
                    )


                with r3:

                    if confidence is not None:

                        st.metric(
                            "AI Confidence",
                            f"{confidence:.2f}%"
                        )

                    else:

                        st.metric(
                            "AI Confidence",
                            "N/A"
                        )


                st.write("")


                # Threat level
                if (
                    str(predicted_attack).lower()
                    == "normal"
                ):

                    st.markdown(
                        """
                        <div class="threat-low">
                            <div class="threat-title">
                                🟢 LOW THREAT
                            </div>
                            <div>
                                Normal network traffic detected.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    recommendation = (
                        "No immediate action required. "
                        "Continue normal network monitoring."
                    )


                elif (
                    str(actual_attack).lower()
                    == "attack"
                ):

                    st.markdown(
                        """
                        <div class="threat-high">
                            <div class="threat-title">
                                🔴 HIGH THREAT
                            </div>
                            <div>
                                Potential network intrusion detected.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    recommendation = (
                        "Investigate the source of the network "
                        "traffic, monitor the affected connection "
                        "and isolate the suspicious host if "
                        "malicious activity is confirmed."
                    )


                else:

                    st.markdown(
                        """
                        <div class="threat-medium">
                            <div class="threat-title">
                                🟡 MEDIUM THREAT
                            </div>
                            <div>
                                Suspicious network activity detected.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    recommendation = (
                        "Monitor the network activity and "
                        "perform additional security analysis."
                    )


                st.write("")


                # Recommendation
                st.markdown(
                    "### 🛡️ Security Recommendation"
                )

                st.info(
                    recommendation
                )


            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


    except Exception as e:

        st.error(
            f"Test dataset could not be loaded: {e}"
        )


# =========================================================
# TAB 4 — ABOUT PROJECT
# =========================================================

with tab4:

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )


    about1, about2 = st.columns(2)


    with about1:

        st.markdown(
            """
            ### 🎯 Objective

            This project applies Artificial Intelligence
            and Machine Learning techniques to identify
            suspicious network traffic and potential
            cybersecurity threats.

            ### 📂 Dataset

            **NSL-KDD**

            The dataset contains network traffic features
            representing normal and different types of
            attacks.

            ### 🤖 Algorithm

            **Random Forest Classifier**

            A supervised machine learning algorithm
            that combines multiple decision trees for
            classification.
            """
        )


    with about2:

        st.markdown(
            """
            ### 🛠️ Technologies

            - Python
            - Pandas
            - NumPy
            - Scikit-learn
            - Matplotlib
            - Joblib
            - Streamlit

            ### 🔄 Workflow

            Dataset → Preprocessing → Encoding →
            Random Forest → Evaluation →
            Prediction → Threat Detection

            ### 🚀 Future Scope

            - Real-time monitoring
            - Automated alerts
            - Firewall integration
            - Cloud deployment
            - Deep learning
            """
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <center>
    <span style="color:#64748b;">
    🔐 AI-Based Cybersecurity Network Threat & Intrusion Profiler
    </span>
    <br>
    <span style="color:#475569; font-size:12px;">
    IBM Internship Project • Artificial Intelligence & Machine Learning
    </span>
    </center>
    """,
    unsafe_allow_html=True
)