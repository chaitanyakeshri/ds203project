import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import joblib

# Load preprocessed data and models
df = joblib.load("df_with_vectors.pkl")  # Must include: 'filtered_tokens', 'vector', 'w2v_cluster', 'SerialNo'
w2v_model = joblib.load("w2v_model.pkl")
global_cluster_vectors = joblib.load("global_cluster_vectors.pkl")  # {cluster_id: vector}

st.set_page_config(page_title="Summary Explorer", layout="centered")

st.title("📚 Summary Explorer")
st.markdown("Input keywords to find the most relevant session and top summaries.")

# Input box
keyword_input = st.text_input("Enter keywords (comma-separated):", "")

if keyword_input:
    keywords = [word.strip().lower() for word in keyword_input.split(",")]
    keyword_vectors = []

    for word in keywords:
        if word in w2v_model.wv:
            keyword_vectors.append(w2v_model.wv[word])

    if not keyword_vectors:
        st.warning("None of the entered keywords were found in the vocabulary.")
    else:
        # Get average keyword vector
        keyword_avg = np.mean(keyword_vectors, axis=0)

        # Compare with cluster vectors
        cluster_scores = []
        for cluster_id, vector in global_cluster_vectors.items():
            if np.linalg.norm(vector) != 0:
                similarity = 1 - cosine(keyword_avg, vector)
            else:
                similarity = 0
            cluster_scores.append((cluster_id, similarity))

        # Get best matching cluster
        best_cluster, best_score = max(cluster_scores, key=lambda x: x[1])

        st.subheader(f"🔍 Most Relevant Session: Cluster {best_cluster} (Score: {best_score:.4f})")

        # Get top 3 summaries from this cluster
        cluster_df = df[df['w2v_cluster'] == best_cluster][['SerialNo', 'vector', 'Session_Summary']]
        cluster_df['similarity'] = cluster_df['vector'].apply(
            lambda vec: 1 - cosine(vec, keyword_avg) if np.linalg.norm(vec) != 0 and np.linalg.norm(keyword_avg) != 0 else 0
        )
        top_summaries = cluster_df.sort_values(by='similarity', ascending=False).head(3)

        st.markdown("### 📝 Top 3 Relevant Summaries:")
        for i, row in top_summaries.iterrows():
            st.markdown(f"**Serial No: {row['SerialNo']}**")
            st.markdown(f"> {row['Session_Summary']}")
            st.markdown("---")
