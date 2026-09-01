import streamlit as st
import sys
from collections import Counter
import pandas as pd

# Add src folder
sys.path.append("src")

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


/* Main title */

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #123B5D;
    margin-bottom: 5px;
}


/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #496579;
    margin-bottom: 30px;
}


/* Cards */

.card {
    background: rgba(255, 255, 255, 0.88);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
    border: 1px solid #D7E5E8;
    box-shadow: 0 5px 18px rgba(50, 80, 90, 0.10);
}


/* Section titles */

.section-title {
    font-size: 27px;
    font-weight: 700;
    color: #123B5D;
    margin-top: 20px;
}


/* Metric values */

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #123B5D;
}


/* DNA sequence box */

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


/* Low risk */

.result-low {
    background: #E8F7F0;
    color: #126B4A;
    border: 2px solid #8DD8BA;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}


/* Medium risk */

.result-medium {
    background: #FFF6DD;
    color: #806000;
    border: 2px solid #E8C96A;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}


/* High risk */

.result-high {
    background: #FDECEC;
    color: #9B3030;
    border: 2px solid #E5A0A0;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}


/* Footer */

.footer {
    text-align: center;
    color: #627887;
    padding: 30px;
    font-size: 14px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #EEF6F7;
}


/* Text area */

textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #C8D9DE !important;
}


/* Input labels */

label {
    color: #23445C !important;
    font-weight: 600 !important;
}


/* Buttons */

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


/* Metric cards */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D8E5E8;
    padding: 15px;
    border-radius: 14px;
    box-shadow: 0 3px 10px rgba(50, 80, 90, 0.08);
}


/* General text */

p {
    color: #263F52;
}

</style>
""", unsafe_allow_html=True)
# =========================================================
# FUNCTIONS
# =========================================================

def clean_sequence(sequence):

    sequence = sequence.upper()

    sequence = "".join(
        sequence.split()
    )

    return sequence


def calculate_dna_features(sequence):

    length = len(sequence)

    counts = Counter(sequence)

    num_a = counts.get("A", 0)
    num_t = counts.get("T", 0)
    num_c = counts.get("C", 0)
    num_g = counts.get("G", 0)

    gc_content = (
        (num_g + num_c) / length
    ) * 100

    at_content = (
        (num_a + num_t) / length
    ) * 100

    # 3-mer calculation
    kmers = []

    for i in range(length - 2):

        kmers.append(
            sequence[i:i + 3]
        )

    if kmers:

        kmer_counts = Counter(kmers)

        most_common_kmer = (
            kmer_counts.most_common(1)[0][0]
        )

        most_common_count = (
            kmer_counts.most_common(1)[0][1]
        )

        kmer_3_freq = (
            most_common_count /
            len(kmers)
        )

    else:

        most_common_kmer = "N/A"

        kmer_3_freq = 0


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
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧬 Digital DNA Score</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'DNA Sequence Analysis & Disease Risk Prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧬 DNA Analyzer")

    st.write(
        "Enter your DNA sequence and analyze "
        "its characteristics."
    )

    mutation_flag = st.selectbox(
        "Mutation Flag",
        ["No", "Yes"]
    )

    st.markdown("---")

    st.info(
        "This dashboard uses a machine-learning "
        "model trained on synthetic DNA data."
    )


# =========================================================
# INPUT
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🔬 DNA Sequence Input</div>',
    unsafe_allow_html=True
)

sequence = st.text_area(
    "Paste your DNA sequence",
    height=160,
    placeholder=(
        "Example:\n"
        "ATGCGTACGTTAGCGATCGATCGTAGCTAGCTAGGCTAACG"
    ),
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🧬 ANALYZE DNA",
    use_container_width=True
)


if analyze:

    sequence = clean_sequence(sequence)


    # =====================================================
    # VALIDATION
    # =====================================================

    if not sequence:

        st.error(
            "Please enter a DNA sequence."
        )

        st.stop()


    valid_characters = set("ATCG")

    invalid_characters = (
        set(sequence) -
        valid_characters
    )


    if invalid_characters:

        st.error(
            "Invalid DNA sequence. "
            "Only A, T, C and G are allowed."
        )

        st.stop()


    if len(sequence) < 3:

        st.error(
            "Sequence must contain at least 3 bases."
        )

        st.stop()


    # =====================================================
    # CALCULATE FEATURES
    # =====================================================

    features = calculate_dna_features(
        sequence
    )


    # =====================================================
    # DNA OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 DNA Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Sequence Length",
            features["sequence_length"]
        )


    with col2:

        st.metric(
            "GC Content",
            f"{features['gc_content']:.2f}%"
        )


    with col3:

        st.metric(
            "AT Content",
            f"{features['at_content']:.2f}%"
        )


    with col4:

        st.metric(
            "3-mer Frequency",
            f"{features['kmer_3_freq']:.3f}"
        )


    # =====================================================
    # BASE COMPOSITION
    # =====================================================

    st.markdown(
        '<div class="section-title">🧪 Base Composition</div>',
        unsafe_allow_html=True
    )


    base_data = pd.DataFrame({

        "Base": [
            "A",
            "T",
            "C",
            "G"
        ],

        "Count": [

            features["num_a"],

            features["num_t"],

            features["num_c"],

            features["num_g"]

        ]
    })


    st.bar_chart(
        base_data.set_index("Base")
    )


    # =====================================================
    # GC / AT VISUALIZATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🧬 GC / AT Distribution</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"GC Content: "
            f"**{features['gc_content']:.2f}%**"
        )

        st.progress(
            min(
                features["gc_content"] / 100,
                1.0
            )
        )


    with col2:

        st.write(
            f"AT Content: "
            f"**{features['at_content']:.2f}%**"
        )

        st.progress(
            min(
                features["at_content"] / 100,
                1.0
            )
        )


    # =====================================================
    # DNA SEQUENCE DISPLAY
    # =====================================================

    st.markdown(
        '<div class="section-title">🧬 Sequence Visualization</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="dna-box">
        {sequence}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # MUTATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🔬 Mutation Analysis</div>',
        unsafe_allow_html=True
    )


    if mutation_flag == "Yes":

        st.warning(
            "⚠️ Mutation flag detected."
        )

    else:

        st.success(
            "✅ No mutation flag detected."
        )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        prediction, probabilities = (
            predict_disease_risk(

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
        )


        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Disease Risk Prediction'
            '</div>',
            unsafe_allow_html=True
        )


        if prediction == "Low":

            st.markdown(
                f"""
                <div class="result-low">
                    <h1>🟢 LOW RISK</h1>
                    <h3>Prediction: {prediction}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )


        elif prediction == "Medium":

            st.markdown(
                f"""
                <div class="result-medium">
                    <h1>🟡 MEDIUM RISK</h1>
                    <h3>Prediction: {prediction}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            st.markdown(
                f"""
                <div class="result-high">
                    <h1>🔴 HIGH RISK</h1>
                    <h3>Prediction: {prediction}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📈 Prediction Probability'
            '</div>',
            unsafe_allow_html=True
        )


        for risk, probability in probabilities.items():

            percentage = (
                probability * 100
            )

            st.write(
                f"**{risk}: {percentage:.2f}%**"
            )

            st.progress(
                float(probability)
            )


        # =================================================
        # EXTRA INFORMATION
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🔎 DNA Insights'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.info(
                f"Most common 3-mer: "
                f"**{features['most_common_kmer']}**"
            )


        with col2:

            st.info(
                f"A + T bases: "
                f"**{features['num_a'] + features['num_t']}**"
            )


        with col3:

            st.info(
                f"C + G bases: "
                f"**{features['num_c'] + features['num_g']}**"
            )


        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        report = pd.DataFrame({

            "Feature": [

                "Sequence Length",

                "GC Content",

                "AT Content",

                "A Count",

                "T Count",

                "C Count",

                "G Count",

                "3-mer Frequency",

                "Mutation Flag",

                "Predicted Risk"

            ],

            "Value": [

                features["sequence_length"],

                f"{features['gc_content']:.2f}%",

                f"{features['at_content']:.2f}%",

                features["num_a"],

                features["num_t"],

                features["num_c"],

                features["num_g"],

                features["kmer_3_freq"],

                mutation_flag,

                prediction

            ]
        })


        csv = report.to_csv(
            index=False
        )


        st.download_button(

            label="📥 Download DNA Analysis Report",

            data=csv,

            file_name="dna_analysis_report.csv",

            mime="text/csv",

            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    🧬 Digital DNA Score | Machine Learning Project<br>
    Built using Python, Pandas, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
