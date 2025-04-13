import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import joblib

# Load preprocessed data and models
df = joblib.load("df_with_vectors.pkl")  # Must include: 'filtered_tokens', 'vector', 'w2v_cluster', 'SerialNo'
w2v_model = joblib.load("w2v_model.pkl")
global_cluster_vectors = joblib.load("global_cluster_vectors.pkl")  # {cluster_id: vector}

st.set_page_config(page_title="Summary Explorer", layout="wide")

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

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import random
import joblib

session_keywords_dict = joblib.load("session_keywords_dict.pkl")

st.title("🧠 Session Keyword Explorer")

st.markdown("### 📊 Session Bubble Chart (Bubble size = number of keywords)")

import matplotlib.pyplot as plt
import numpy as np

# Setup
session_ids = list(session_keywords_dict.keys())
keyword_counts = [len(keywords) for keywords in session_keywords_dict.values()]
sizes = [count * 150 for count in keyword_counts]  # Bubble area
radii = [np.sqrt(size / np.pi) / 100 for size in sizes]  # Convert to approximate radius in normalized units

# Bubble placement
positions = []
rng = np.random.default_rng(36)
max_attempts = 1000

for i in range(len(session_ids)):
    placed = False
    attempts = 0
    while not placed and attempts < max_attempts:
        x, y = rng.random(), rng.random()
        overlap = False
        for (px, py), pr in zip(positions, radii[:len(positions)]):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < (radii[i] + pr + 0.03):  # 0.03 buffer
                overlap = True
                break
        if not overlap:
            positions.append((x, y))
            placed = True
        attempts += 1
    if not placed:
        positions.append((x, y))  # fallback

# Unpack positions
x_vals, y_vals = zip(*positions)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(x_vals, y_vals, s=sizes, alpha=0.8, c=keyword_counts,
                     cmap='rainbow', edgecolors='black', linewidth=1, zorder=2)

# Add session IDs
for i, session_id in enumerate(session_ids):
    ax.text(x_vals[i], y_vals[i], str(session_id), ha='center', va='center',
            fontsize=9, color='white', weight='bold', zorder=3)

# Style
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, which='both', linestyle='--', linewidth=0.6, color='gray')
ax.set_title("Sessions as Bubbles", fontsize=16)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

st.pyplot(fig)


# Session selector
selected_session = st.selectbox("🔍 Select a session to view its keyword word clouds:", session_ids)

# Word frequency in the selected session
selected_keywords = session_keywords_dict[selected_session]
session_freq = pd.Series(selected_keywords).value_counts().to_dict()

# Global keyword frequency
all_keywords = sum(session_keywords_dict.values(), [])
global_freq = pd.Series(all_keywords).value_counts().to_dict()

# Word Clouds
col1, col2 = st.columns([1, 1])  # equal width columns

with col1:
    st.markdown("#### 🌐 Session-specific Keyword Importance")
    wc = WordCloud(width=800, height=500, background_color='white', colormap='viridis')
    wc.generate_from_frequencies(session_freq)
    st.image(wc.to_image())  # don't use use_column_width=True to preserve size

with col2:
    st.markdown("#### 🌍 Global Keyword Importance")
    wc_global = WordCloud(width=800, height=500, background_color='white', colormap='plasma')
    wc_global.generate_from_frequencies(global_freq)
    st.image(wc_global.to_image())  # don't use use_column_width=True to preserve size

