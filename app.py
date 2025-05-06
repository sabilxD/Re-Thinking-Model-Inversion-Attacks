import streamlit as st
import os
import cv2
from PIL import Image

# ------------------- CONFIG -------------------
OUTPUT_ROOT = "cropped_faces"
IMAGE_SIZE = (128, 128)

# ------------------- LOAD -------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.resize(img, IMAGE_SIZE))

# ------------------- DISPLAY -------------------
def display_face_pairs_grid(folder_path):
    valid_pairs = []
    gen_files = sorted([f for f in os.listdir(folder_path) if f.startswith("gen_")])

    for f in gen_files:
        idx = f.split("_")[1].split(".")[0]
        gen_path = os.path.join(folder_path, f"gen_{idx}.png")
        best_path = os.path.join(folder_path, f"best_{idx}.png")

        if os.path.exists(best_path):
            valid_pairs.append((gen_path, best_path, idx))

    if not valid_pairs:
        st.info("No valid image pairs found in this folder.")
        return

    for i in range(0, len(valid_pairs), 2):
        cols = st.columns(4)

        # First pair
        if i < len(valid_pairs):
            gen_img = load_image(valid_pairs[i][0])
            best_img = load_image(valid_pairs[i][1])
            idx = valid_pairs[i][2]

            with cols[0]:
                st.image(gen_img, use_column_width=True)
                st.markdown(f"<div style='text-align:center; font-size:18px;'>🧪 Generated {idx}</div>", unsafe_allow_html=True)

            with cols[1]:
                st.image(best_img, use_column_width=True)
                st.markdown(f"<div style='text-align:center; font-size:18px;'>🎯 Real Match {idx}</div>", unsafe_allow_html=True)

        # Second pair
        if i + 1 < len(valid_pairs):
            gen_img = load_image(valid_pairs[i+1][0])
            best_img = load_image(valid_pairs[i+1][1])
            idx = valid_pairs[i+1][2]

            with cols[2]:
                st.image(gen_img, use_column_width=True)
                st.markdown(f"<div style='text-align:center; font-size:18px;'>🧪 Generated {idx}</div>", unsafe_allow_html=True)

            with cols[3]:
                st.image(best_img, use_column_width=True)
                st.markdown(f"<div style='text-align:center; font-size:18px;'>🎯 Real Match {idx}</div>", unsafe_allow_html=True)

        st.markdown("---")


# ------------------- STYLING -------------------
def local_css():
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stImage > img {
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        }
        .element-container p {
            font-size: 1.1rem !important;
            font-weight: 600;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------- MAIN APP -------------------
st.set_page_config(page_title="Model Inversion Visualizer", layout="wide")
local_css()

st.title("🔍 Model Inversion Visualizer")
st.markdown("Visual comparison between **Generated Images** and their **Closest Matches** from LFW.")

# Get folders inside OUTPUT_ROOT
batch_folders = sorted([f for f in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, f))])
if not batch_folders:
    st.warning("No batch folders found in the output directory.")
else:
    selected_batch = st.sidebar.selectbox("📁 Select a Folder", batch_folders)
    selected_path = os.path.join(OUTPUT_ROOT, selected_batch)

    if os.path.exists(selected_path):
        display_face_pairs_grid(selected_path)
    else:
        st.warning(f"No valid data found for {selected_batch}")
