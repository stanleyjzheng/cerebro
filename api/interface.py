import streamlit as st
import requests
from io import BytesIO

# --- Constants ---
API_BASE = "http://localhost:8000"
DEFAULT_TOP_N = 5

# --- API Helper Functions ---
def fetch_text_results(query: str, top_n: int = DEFAULT_TOP_N, filter_media: str = None):
    params = {"query": query, "top_n": top_n}
    if filter_media:
        params["filter_media"] = filter_media
    response = requests.get(f"{API_BASE}/search/text", params=params)
    response.raise_for_status()
    return response.json()

def fetch_image_results(file_bytes: bytes, filename: str, top_n: int = DEFAULT_TOP_N, filter_media: str = None):
    files = {"file": (filename, BytesIO(file_bytes))}
    data = {"top_n": top_n}
    if filter_media:
        data["filter_media"] = filter_media
    response = requests.post(f"{API_BASE}/search/image", files=files, data=data)
    response.raise_for_status()
    return response.json()

def display_results(results):
    if not results:
        st.info("No results found.")
        return

    st.markdown("---")
    st.subheader("Results")
    # Display results in a 3-column grid.
    cols = st.columns(3)
    for idx, item in enumerate(results):
        with cols[idx % 3]:
            if item.get("media_type") == "image" and item.get("file_path"):
                st.image(item["file_path"], use_container_width=True)
            if item.get("youtube_video_id"):
                yt_url = f"https://www.youtube.com/watch?v={item['youtube_video_id']}"
                st.markdown(f'<a href="{yt_url}" target="_blank">Watch on YouTube</a>', unsafe_allow_html=True)
            st.caption(f"Score: {item.get('score', 0):.2f}")
            if item.get("timestamp"):
                st.caption(f"Timestamp: {item['timestamp']}")

# --- Callback for Text Search ---
def on_input_change():
    # Every keystroke triggers this callback.
    query = st.session_state.user_input  # May be empty.
    with st.spinner("Searching..."):
        st.text(f"Found {len(st.session_state.user_input)} results.")
        try:
            results = fetch_text_results(query, DEFAULT_TOP_N, st.session_state.filter_val)
            st.session_state.results = results
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.results = []

# --- Page Setup ---
st.set_page_config(page_title="Multi-Modal Search", layout="wide")
st.markdown("# Multi-Modal Search")
st.markdown("Search our image and video database using either text or an image query.")

# --- Sidebar: Search Mode & Filter ---
with st.sidebar:
    st.header("Search Settings")
    search_mode = st.radio("Select Search Mode", options=["Text Search", "Image Search"], index=0)
    media_option = st.selectbox("Media Filter", options=["All", "Images", "Videos"])
    # Map the selection to API filter parameters.
    filter_mapping = {"All": None, "Images": "image", "Videos": "youtube"}
    st.session_state.filter_val = filter_mapping[media_option]

# --- Main Content ---
if search_mode == "Text Search":
    st.subheader("Enter your query")
    # This text_input calls on_input_change after every keystroke.
    st.text_input("Type your query here", key="user_input", on_change=on_input_change)
    if "results" in st.session_state:
        display_results(st.session_state.results)

elif search_mode == "Image Search":
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        with st.spinner("Searching..."):
            try:
                results = fetch_image_results(file_bytes, uploaded_file.name, DEFAULT_TOP_N, st.session_state.filter_val)
                st.session_state.results = results
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.results = []
        display_results(st.session_state.results)
