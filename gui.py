import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from wordcloud import WordCloud
import joblib

# Load data
df = joblib.load("df_with_vectors.pkl")
w2v_model = joblib.load("w2v_model.pkl")
global_cluster_vectors = joblib.load("global_cluster_vectors.pkl")
session_keywords_dict = joblib.load("session_keywords_dict.pkl")

st.set_page_config(page_title="Summary Explorer", layout="wide")

st.title("📚 Summary Explorer")
st.markdown("Input keywords to find the most relevant session and top summaries.")

# --- Keyword Input ---
keyword_input = st.text_input("Enter keywords (comma-separated):", "")

if keyword_input:
    keywords = [word.strip().lower() for word in keyword_input.split(",")]
    keyword_vectors = [w2v_model.wv[word] for word in keywords if word in w2v_model.wv]

    if not keyword_vectors:
        st.warning("None of the entered keywords were found in the vocabulary.")
    else:
        keyword_avg = np.mean(keyword_vectors, axis=0)

        # Find best-matching cluster
        cluster_scores = []
        for cluster_id, vector in global_cluster_vectors.items():
            if np.linalg.norm(vector) != 0:
                similarity = 1 - cosine(keyword_avg, vector)
            else:
                similarity = 0
            cluster_scores.append((cluster_id, similarity))

        best_cluster, best_score = max(cluster_scores, key=lambda x: x[1])

        st.subheader(f"🔍 Most Relevant Session: Cluster {best_cluster} (Score: {best_score:.4f})")

        # Find top 3 summaries in this cluster
        cluster_df = df[df['w2v_cluster'] == best_cluster][['SerialNo', 'vector', 'Session_Summary']]
        cluster_df['similarity'] = cluster_df['vector'].apply(
            lambda vec: 1 - cosine(vec, keyword_avg) if np.linalg.norm(vec) != 0 else 0
        )
        top_summaries = cluster_df.sort_values(by='similarity', ascending=False).head(3)

        with st.expander("📖 View Top 3 Relevant Summaries"):
            for i, row in top_summaries.iterrows():
                st.markdown(f"**Serial No: {row['SerialNo']}**")
                st.markdown(f"> {row['Session_Summary']}")
                st.markdown("---")

# --- Bubble Chart Section ---
st.title("🧠 Session Keyword Explorer")
st.markdown("### 📊 Session Bubble Chart (Bubble size = number of keywords)")

session_ids = list(session_keywords_dict.keys())
keyword_counts = [len(kws) for kws in session_keywords_dict.values()]
sizes = [count * 150 for count in keyword_counts]
radii = [np.sqrt(size / np.pi) / 100 for size in sizes]

# Non-overlapping positions
positions = []
rng = np.random.default_rng(47)
max_attempts = 1000

for i in range(len(session_ids)):
    placed = False
    attempts = 0
    while not placed and attempts < max_attempts:
        x, y = rng.random(), rng.random()
        overlap = any(np.sqrt((x - px)**2 + (y - py)**2) < (radii[i] + pr + 0.03)
                      for (px, py), pr in zip(positions, radii))
        if not overlap:
            positions.append((x, y))
            placed = True
        attempts += 1
    if not placed:
        positions.append((x, y))  # fallback

x_vals, y_vals = zip(*positions)

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(x_vals, y_vals, s=sizes, alpha=0.8, c=keyword_counts,
                     cmap='rainbow', edgecolors='black', linewidth=1, zorder=2)

# Add session IDs
for i, session_id in enumerate(session_ids):
    ax.text(x_vals[i], y_vals[i], str(session_id), ha='center', va='center',
            fontsize=9, color='white', weight='bold', zorder=3)

# Style the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, linestyle='--', linewidth=0.6, color='gray')
ax.set_title("Sessions as Bubbles", fontsize=16)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

st.pyplot(fig)

# --- Word Clouds ---
selected_session = st.selectbox("🔍 Select a session to view its keyword word clouds:", session_ids)

# Word frequency
selected_keywords = session_keywords_dict[selected_session]
session_freq = pd.Series(selected_keywords).value_counts().to_dict()

all_keywords = sum(session_keywords_dict.values(), [])
global_freq = pd.Series(all_keywords).value_counts().to_dict()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 🌐 Session-specific Keyword Importance")
    wc = WordCloud(width=800, height=500, background_color='white', colormap='viridis')
    wc.generate_from_frequencies(session_freq)
    st.image(wc.to_image())

with col2:
    st.markdown("#### 🌍 Global Keyword Importance")
    wc_global = WordCloud(width=800, height=500, background_color='white', colormap='plasma')
    wc_global.generate_from_frequencies(global_freq)
    st.image(wc_global.to_image())
