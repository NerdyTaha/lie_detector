import streamlit as st
import tempfile
import os
import time

# Import our backend modules
from correctness import analyze_correctness
from audio_analyse import analyze_audio_features
from visual_analyse import analyze_visual
from final_decision import generate_final_verdict

# Page Config
st.set_page_config(
    page_title="LLM Lie Detector",
    page_icon="🕵️",
    layout="wide"
)

# Custom CSS for status badges
st.markdown("""
<style>
    .report-card { border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .stAlert { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ LLM-Powered Lie Detector")
st.markdown("### Multi-Modal Deception Analysis System")
st.markdown("This system analyzes **Text**, **Audio**, and **Video** cues using Gemini 2.5 Flash to detect potential deception.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Input Data")
    question = st.text_area(
        "Interrogation Question",
        placeholder="e.g., Did you steal the laptop from the breakroom?",
        help="The exact question asked to the subject in the video."
    )
    
    uploaded_file = st.file_uploader(
        "Upload Video Response", 
        type=["mp4", "mov", "avi", "mkv"]
    )
    
    analyze_btn = st.button("🔍 Run Full Analysis", type="primary")

# --- Main Logic ---

if analyze_btn:
    if not question or not uploaded_file:
        st.error("⚠️ Please provide both a Question and a Video file.")
    else:
        # Create a placeholder for logs/progress
        status_container = st.container()
        
        # 1. Save uploaded file to a temporary path
        # We do this ONCE so all modules access the same file path on disk
        # avoiding read-pointer issues with file-like objects.
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()

        try:
            # --- PHASE 1: TEXTUAL ANALYSIS ---
            with status_container:
                with st.spinner("Phase 1/4: Extracting audio & analyzing content consistency..."):
                    text_result = analyze_correctness(video_path, question)
            
            # --- PHASE 2: AUDIO ANALYSIS ---
            with status_container:
                with st.spinner("Phase 2/4: Analyzing vocal pitch, pauses, and tremors..."):
                    audio_result = analyze_audio_features(video_path, question)

            # --- PHASE 3: VISUAL ANALYSIS ---
            with status_container:
                with st.spinner("Phase 3/4: Extracting frames & analyzing micro-expressions..."):
                    visual_result = analyze_visual(video_path, question, num_frames=8)

            # --- PHASE 4: FINAL VERDICT ---
            with status_container:
                with st.spinner("Phase 4/4: Synthesizing final verdict..."):
                    final_verdict = generate_final_verdict(
                        question, 
                        text_result, 
                        audio_result, 
                        visual_result
                    )
            
            # --- DISPLAY RESULTS ---
            st.divider()
            
            # 1. Final Verdict Section (Top)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Final Verdict")
                if "LIE" in final_verdict.verdict.value:
                    st.error(f"## {final_verdict.verdict.value}")
                elif "TRUTHFUL" in final_verdict.verdict.value:
                    st.success(f"## {final_verdict.verdict.value}")
                else:
                    st.warning(f"## {final_verdict.verdict.value}")
                
                st.metric("Confidence Score", f"{final_verdict.confidence_score}%")
                
            with col2:
                st.subheader("Investigator's Synthesis")
                st.info(final_verdict.synthesis_reasoning)

            st.divider()
            st.subheader("Detailed Evidence Breakdown")

            # 2. Detailed Reports (Columns)
            c1, c2, c3 = st.columns(3)

            # Helper to color code signals
            def get_color(signal):
                return "red" if signal == "positive" else "green"

            # Text Report
            with c1:
                st.markdown("#### 📝 Content Analysis")
                color = get_color(text_result.signal)
                st.markdown(f"Signal: :{color}[**{text_result.signal.upper()}**]")
                with st.expander("View Reasoning"):
                    st.write(text_result.reasoning)

            # Audio Report
            with c2:
                st.markdown("#### 🔊 Audio Analysis")
                color = get_color(audio_result.signal)
                st.markdown(f"Signal: :{color}[**{audio_result.signal.upper()}**]")
                with st.expander("View Reasoning"):
                    st.write(audio_result.reasoning)

            # Visual Report
            with c3:
                st.markdown("#### 👁️ Visual Analysis")
                color = get_color(visual_result.signal)
                st.markdown(f"Signal: :{color}[**{visual_result.signal.upper()}**]")
                with st.expander("View Reasoning"):
                    st.write(visual_result.reasoning)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Optionally print full traceback to console
            print(e)
        
        finally:
            # Cleanup the main temp video file
            if os.path.exists(video_path):
                os.remove(video_path)

elif uploaded_file is not None:
    # Just show video preview if not analyzing yet
    st.video(uploaded_file)