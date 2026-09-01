import os
import sys
from collections import Counter
import pandas as pd
import streamlit as st

# Dynamically add 'src' directory to Python path BEFORE importing from it
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_disease_risk


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Digital DNA Score",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F4F9F9 0%,
        #EAF4F4 50%,
        #F8F5EF 100%
    );
    color: #172B4D;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #123B5D;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #496579;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.88);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    border: 1px solid #D7E5E8;
    box-shadow: 0 5px 18px rgba(50, 80, 90, 0.10);
}

.section-title {
    font-size: 27px;
    font-weight: 700;
    color: #123B5D;
    margin-top: 20px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #123B5D;
}

.dna-box {
    background: #FFFFFF;
    color: #17324D;
    border-radius: 15px;
    padding: 18px;
    font-family: monospace;
    line-height: 1.8;
    word-break: break-all;
    border: 1px solid #D5E2E6;
    box-shadow: 0 4px 12px rgba(50, 80, 90, 0.08);
}

.result-low {
    background: #E8F7F0;
    color: #126B4A;
    border: 2px solid #8DD8BA;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-medium {
    background: #FFF6DD;
    color: #806000;
    border: 2px solid #E8C96A;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-high {
    background: #FDECEC;
    color: #9B3030;
    border: 2px solid #E5A0A0;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #627887;
    padding: 30px;
    font-size: 14px;
}

section[data-testid="stSidebar"] {
    background: #EEF6F7;
}

textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #C8D9DE !important;
}

label {
    color: #23445C !important;
    font-weight: 600 !important;
}

.stButton > button {
    background-color: #3A7D7C;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    padding: 12px;
}

.stButton > button:hover {
    background-color: #2F6665;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D8E5E8;
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0 3px 10px rgba(50, 80, 90, 0.08);
}

p {
    color: #263F52;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_sequence(seq):
    return "".join(seq.upper().split())


def calculate_dna_features(seq):
    length = len(seq)
    counts = Counter(seq)

    num_a = counts.get("A", 0)
    num_t = counts.get("T", 0)
    num_c = counts.get("C", 0)
    num_g = counts.get("G", 0)

    gc_content = ((num_g + num_c) / length) * 100 if length > 0 else 0
    at_content = ((num_a + num_t) / length) * 100 if length > 0 else 0

    kmers = [seq[i:i + 3] for i in range(length - 2)]

    if kmers:
        kmer_counts = Counter(kmers)
        most_common_kmer, most_common_count = kmer_counts.most_common(1)[0]
        kmer_3_freq = most_common_count / len(kmers)
    else:
        most_common_kmer = "N/A"
        kmer_3_freq = 0.0

    return {
        "gc_content": gc_content,
        "at_content": at_content,
        "sequence_length": length,
        "num_a": num_a,
        "num_t": num_t,
        "num_c": num_c,
        "num_g": num_g,
        "kmer_3_freq": kmer_3_freq,
        "most_common_kmer": most_common_kmer
    }


# =========================================================
# HEADER & SIDEBAR
# =========================================================

st.markdown('<div class="main-title">🧬 Digital DNA Score</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">DNA Sequence Analysis & Disease Risk Prediction</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🧬 DNA Analyzer")
    st.write("Enter your DNA sequence and analyze its characteristics.")
    mutation_flag = st.selectbox("Mutation Flag", ["No", "Yes"])
    st.markdown("---")
    st.info("This dashboard uses a machine-learning model trained on synthetic DNA data.")


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔬 DNA Sequence Input</div>', unsafe_allow_html=True)

raw_sequence = st.text_area(
    "Paste your DNA sequence",
    height=160,
    placeholder="Example:\nATGCGTACGTTAGCGATCGATCGTAGCTAGCTAGGCTAACG",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ANALYZE BUTTON & RESULTS
# =========================================================

analyze = st.button("🧬 ANALYZE DNA", use_container_width=True)

if analyze:
    sequence = clean_sequence(raw_sequence)

    if not sequence:
        st.error("Please enter a DNA sequence.")
        st.stop()

    invalid_chars = set(sequence) - set("ATCG")
    if invalid_chars:
        st.error(f"Invalid DNA sequence. Allowed bases are A, T, C, G. Invalid found: {', '.join(invalid_chars)}")
        st.stop()

    if len(sequence) < 3:
        st.error("Sequence must contain at least 3 bases.")
        st.stop()

    features = calculate_dna_features(sequence)

    # Overview Metrics
    st.markdown('<div class="section-title">📊 DNA Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sequence Length", features["sequence_length"])
    c2.metric("GC Content", f"{features['gc_content']:.2f}%")
    c3.metric("AT Content", f"{features['at_content']:.2f}%")
    c4.metric("3-mer Frequency", f"{features['kmer_3_freq']:.3f}")

    # Base Composition
    st.markdown('<div class="section-title">🧪 Base Composition</div>', unsafe_allow_html=True)
    base_df = pd.DataFrame({
        "Base": ["A", "T", "C", "G"],
        "Count": [features["num_a"], features["num_t"], features["num_c"], features["num_g"]]
    })
    st.bar_chart(base_df.set_index("Base"))

    # Distribution Progress
    st.markdown('<div class="section-title">🧬 GC / AT Distribution</div>', unsafe_allow_html=True)
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.write(f"GC Content: **{features['gc_content']:.2f}%**")
        st.progress(min(features["gc_content"] / 100, 1.0))
    with pcol2:
        st.write(f"AT Content: **{features['at_content']:.2f}%**")
        st.progress(min(features["at_content"] / 100, 1.0))

    # Sequence Display
    st.markdown('<div class="section-title">🧬 Sequence Visualization</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dna-box">{sequence}</div>', unsafe_allow_html=True)

    # Mutation Analysis
    st.markdown('<div class="section-title">🔬 Mutation Analysis</div>', unsafe_allow_html=True)
    if mutation_flag == "Yes":
        st.warning("⚠️ Mutation flag detected.")
    else:
        st.success("✅ No mutation flag detected.")

    # Model Prediction
    try:
        prediction, probabilities = predict_disease_risk(
            gc_content=features["gc_content"],
            at_content=features["at_content"],
            sequence_length=features["sequence_length"],
            num_a=features["num_a"],
            num_t=features["num_t"],
            num_c=features["num_c"],
            num_g=features["num_g"],
            kmer_3_freq=features["kmer_3_freq"],
            mutation_flag=mutation_flag
        )

        st.markdown('<div class="section-title">🎯 Disease Risk Prediction</div>', unsafe_allow_html=True)

        if prediction == "Low":
            st.markdown(f'<div class="result-low"><h1>🟢 LOW RISK</h1><h3>Prediction: {prediction}</h3></div>', unsafe_allow_html=True)
        elif prediction == "Medium":
            st.markdown(f'<div class="result-medium"><h1>🟡 MEDIUM RISK</h1><h3>Prediction: {prediction}</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-high"><h1>🔴 HIGH RISK</h1><h3>Prediction: {prediction}</h3></div>', unsafe_allow_html=True)

        # Probabilities
        st.markdown('<div class="section-title">📈 Prediction Probability</div>', unsafe_allow_html=True)
        for risk_cls, prob in probabilities.items():
            st.write(f"**{risk_cls}: {prob * 100:.2f}%**")
            st.progress(float(prob))

        # Insights
        st.markdown('<div class="section-title">🔎 DNA Insights</div>', unsafe_allow_html=True)
        i1, i2, i3 = st.columns(3)
        i1.info(f"Most common 3-mer: **{features['most_common_kmer']}**")
        i2.info(f"A + T bases: **{features['num_a'] + features['num_t']}**")
        i3.info(f"C + G bases: **{features['num_c'] + features['num_g']}**")

        # Download CSV
        report = pd.DataFrame({
            "Feature": [
                "Sequence Length", "GC Content", "AT Content", "A Count",
                "T Count", "C Count", "G Count", "3-mer Frequency",
                "Mutation Flag", "Predicted Risk"
            ],
            "Value": [
                features["sequence_length"], f"{features['gc_content']:.2f}%",
                f"{features['at_content']:.2f}%", features["num_a"],
                features["num_t"], features["num_c"], features["num_g"],
                features["kmer_3_freq"], mutation_flag, prediction
            ]
        })
        
        st.download_button(
            label="📥 Download DNA Analysis Report",
            data=report.to_csv(index=False),
            file_name="dna_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
🧬 Digital DNA Score | Machine Learning Project<br>
Built using Python, Pandas, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
