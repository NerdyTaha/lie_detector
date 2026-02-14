import streamlit as st

st.set_page_config(
    page_title="LLM Lie Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ LLM-Powered Lie Detector")
st.markdown(
    "Upload a video response to a question, and the system will analyze whether the answer may be deceptive."
)

# Question input
question = st.text_input(
    "Enter the interrogation question:",
    placeholder="e.g., Did you authorize this transaction?"
)

# Video upload
video_file = st.file_uploader(
    "Upload video response:",
    type=["mp4", "mov", "avi", "mkv"]
)

# Preview video if uploaded
if video_file is not None:
    st.video(video_file)

# Analyze button (no backend yet)
if st.button("Analyze"):
    if not question:
        st.warning("Please enter a question.")
    elif not video_file:
        st.warning("Please upload a video.")
    else:
        st.success("Video and question received. (Backend not integrated yet)")
        st.info("Analysis will appear here once backend is connected.")
