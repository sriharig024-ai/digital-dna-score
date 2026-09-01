import os
import joblib
import pandas as pd
import numpy as np

# Resolve path to the model file saved in src/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def load_model():
    """Loads trained model if present; returns None otherwise."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

_MODEL = load_model()

def predict_disease_risk(gc_content, at_content, sequence_length, num_a, num_t, num_c, num_g, kmer_3_freq, mutation_flag):
    """
    Predicts disease risk class ('Low', 'Medium', 'High') and individual class probabilities.
    """
    mutation_encoded = 1 if mutation_flag == "Yes" else 0
    
    # Construct input dataframe with exact feature names expected by the model
    features_df = pd.DataFrame([{
        "gc_content": float(gc_content),
        "at_content": float(at_content),
        "sequence_length": float(sequence_length),
        "num_a": float(num_a),
        "num_t": float(num_t),
        "num_c": float(num_c),
        "num_g": float(num_g),
        "kmer_3_freq": float(kmer_3_freq),
        "mutation_flag": float(mutation_encoded)
    }])
    
    classes = ["Low", "Medium", "High"]

    # Use trained model if available
    if _MODEL is not None:
        prediction_idx = _MODEL.predict(features_df)[0]
        if hasattr(_MODEL, "predict_proba"):
            probs = _MODEL.predict_proba(features_df)[0]
            probability_dict = {cls: float(p) for cls, p in zip(_MODEL.classes_, probs)}
        else:
            probability_dict = {cls: (1.0 if cls == prediction_idx else 0.0) for cls in classes}
        
        return str(prediction_idx), probability_dict

    # Rule-Based Heuristic Fallback (Used if model.pkl is not yet generated)
    risk_score = (gc_content * 0.3) + (kmer_3_freq * 100 * 0.4) + (mutation_encoded * 40)
    
    if risk_score > 65:
        prediction = "High"
        probs = {"Low": 0.10, "Medium": 0.25, "High": 0.65}
    elif risk_score > 45:
        prediction = "Medium"
        probs = {"Low": 0.20, "Medium": 0.60, "High": 0.20}
    else:
        prediction = "Low"
        probs = {"Low": 0.70, "Medium": 0.20, "High": 0.10}

    return prediction, probs
