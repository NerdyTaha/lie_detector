"""
Usage:
- As a module: from audio_analyse import analyze_audio_features; result = analyze_audio_features(video_input, question)
  video_input can be:
    - a string path to a video file, e.g. "/path/to/video.mp4"
    - a file-like object (e.g. streamlit uploaded file)
- As a standalone script:
    python audio_analyse.py /path/to/video.mp4 "Did you authorize the transaction?"

Requirements:
- ffmpeg installed
- google-generativeai
- moviepy
"""

import os
import time
import json
import tempfile
import typing as t
import logging
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from moviepy.editor import VideoFileClip, concatenate_audioclips
import google.generativeai as genai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------- Pydantic model ----------
class DeceptionSignal(str, Enum):
    LIE_DETECTED = "positive"
    NO_LIE_DETECTED = "negative"

class AudioAnalysis(BaseModel):
    """Analysis result for audio-based deception detection."""
    signal: DeceptionSignal = Field(
        ...,
        description="Detection result: positive if lie detected, negative if not"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the audio-based detection decision"
    )

# ---------- System prompt ----------
# Note: We do not put {transcribed_text} here because Gemini will "hear" the audio file directly.
AUDIO_ANALYSIS_PROMPT = """You are an expert in forensic audio analysis and vocal deception detection. You will analyze the audio characteristics of a spoken answer to determine likelihood of deception.

YOUR TASK:
Analyze the provided audio for vocal and paralinguistic indicators of deception or truthfulness.

*** IMPORTANT AUDIO FORMAT NOTICE ***
The attached audio file is a COMPOSITE. It consists of the FIRST 10 seconds and LAST 10 seconds of the subject's response joined together.
- If the clip is under 20 seconds, it is the full continuous audio.
- If you hear a sudden "jump," cut, or change in background noise in the middle, this is a technical artifact of the splicing. DO NOT interpret this jump as a sign of deception or editing by the user.
- Focus your analysis on the vocal characteristics (trembling, pitch, speed, hesitation) WITHIN the segments provided.

ANALYZE FOR THESE AUDIO-BASED DECEPTION INDICATORS:

1. VOCAL CONFIDENCE & CERTAINTY:
   - Does the voice convey confidence or uncertainty?
   - Is there vocal trembling or shakiness suggesting nervousness?
   - Does vocal strength match the content being delivered?
   - Is there inappropriate confidence (overcompensating) or excessive hesitation?

2. SPEECH FLUENCY & FLOW:
   - Are there excessive pauses, especially before critical information?
   - Unnatural hesitations mid-sentence or before specific details?
   - Frequent self-interruptions or restarts?
   - Unusual gaps between words that break natural rhythm?

3. FILLER WORDS & VERBAL DISFLUENCIES:
   - Excessive "um", "uh", "er", "like", "you know"
   - IMPORTANT: Distinguish between:
     * Natural speech patterns (occasional fillers are normal)
     * Cognitive load markers (clusters of fillers before lies)
   - Are fillers concentrated before key claims or spread throughout?

4. PACE & RHYTHM ANOMALIES:
   - Sudden changes in speaking speed (rushing through parts)
   - Unnaturally slow delivery (fabricating on the spot)
   - Inconsistent pacing compared to baseline speech
   - Rushed responses suggesting rehearsal or evasion

5. VOCAL PITCH & TONE:
   - Pitch elevation suggesting stress or anxiety
   - Monotone delivery lacking natural variation
   - Pitch changes that don't match content emotion
   - Forced or artificial-sounding intonation

6. STUTTERING & REPETITION:
   - Word or phrase repetition not due to normal speech patterns
   - Stuttering on specific words (especially names, numbers, details)
   - IMPORTANT: Consider that some people naturally stutter - look for:
     * Stuttering on critical details vs. throughout
     * Sudden stuttering in otherwise fluent speech

7. SPEECH CLARITY:
   - Mumbling or unclear articulation of specific parts
   - Voice trailing off at the end of sentences
   - Speaking away from microphone (avoidance behavior)

8. EMOTIONAL CONGRUENCE:
   - Does vocal emotion match the content?
   - Inappropriate laughter or nervousness
   - Flat affect when emotion would be expected
   - Forced emotional display

9. BREATH PATTERNS:
   - Audible sighing or heavy breathing
   - Shallow breathing affecting speech flow
   - Breath-holding before answering

DECISION FRAMEWORK:
- POSITIVE (Lie Detected): Multiple clear indicators clustered around key claims, significant departure from natural speech patterns
- NEGATIVE (No Lie Detected): Natural speech flow with normal variation, or indicators explainable by nervousness/personality

---
QUESTION BEING ANSWERED: {question}

Analyze the audio provided in the attachment and provide your assessment in the required format.
"""

# ---------- Helpers ----------
def _is_path_like(obj) -> bool:
    return isinstance(obj, str)

def _save_filelike_to_tempfile(filelike, suffix=""):
    """Save a file-like object to a temporary file and return path."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(filelike.read())
    return tmp_path

def _to_send_audio_path(out_audio_ext: str = ".mp3") -> str:
    ext = out_audio_ext if out_audio_ext.startswith(".") else f".{out_audio_ext}"
    return os.path.join(ASSETS_DIR, f"to_send_audio{ext}")

def _is_persistent_audio_asset(path: str) -> bool:
    abs_path = os.path.abspath(path)
    assets_path = os.path.abspath(ASSETS_DIR)
    return abs_path.startswith(assets_path + os.sep) and os.path.basename(abs_path).startswith("to_send_audio")

def extract_audio_from_video(video_input: t.Union[str, t.IO], out_audio_ext: str = ".mp3") -> str:
    """
    Accepts a video path or file-like object. 
    Returns path to audio file (mp3).
    Logic:
      - If video < 20s: Extract full audio.
      - If video >= 20s: Extract first 10s + last 10s combined.
    """
    needs_cleanup = []
    try:
        if _is_path_like(video_input):
            video_path = video_input
        else:
            suffix = ""
            name = getattr(video_input, "name", "")
            if name and "." in name:
                suffix = "." + name.split(".")[-1]
            video_path = _save_filelike_to_tempfile(video_input, suffix=suffix)
            needs_cleanup.append(video_path)

        logger.info("Loading video for audio extraction: %s", video_path)
        
        clip = VideoFileClip(video_path)
        
        # --- NEW LOGIC START ---
        if clip.duration and clip.duration > 20.0:
            logger.info(f"Video is {clip.duration:.2f}s. Splicing first 10s and last 10s.")
            # Subclip audio directly
            part1 = clip.audio.subclip(0, 10)
            part2 = clip.audio.subclip(clip.duration - 10, clip.duration)
            final_audio = concatenate_audioclips([part1, part2])
        else:
            logger.info("Video is under 20s. Using full audio.")
            final_audio = clip.audio
        # --- NEW LOGIC END ---

        output_audio_path = _to_send_audio_path(out_audio_ext)
        
        # Write the final (possibly spliced) audio
        final_audio.write_audiofile(output_audio_path)
        
        # Cleanup moviepy resources
        clip.close()
            
        return output_audio_path

    except Exception as e:
        logger.exception("Failed to extract audio: %s", e)
        raise
    finally:
        for p in needs_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass

def _upload_to_gemini(path: str, mime_type: str = "audio/mp3"):
    """Uploads the file to Gemini and waits for it to be active."""
    file = genai.upload_file(path, mime_type=mime_type)
    logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
    
    # Wait for file to be ready
    while file.state.name == "PROCESSING":
        logger.info("Waiting for audio file processing...")
        time.sleep(1)
        file = genai.get_file(file.name)
        
    if file.state.name == "FAILED":
        raise ValueError("Gemini file processing failed.")
        
    logger.info(f"Audio file is {file.state.name}")
    return file

def _call_gemini_with_audio(question: str, audio_path: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Uploads audio to Gemini and gets a structured analysis.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    genai.configure(api_key=api_key)

    gemini_file = None
    try:
        # 1. Upload the audio file
        gemini_file = _upload_to_gemini(audio_path, mime_type="audio/mp3")

        # 2. Configure model with schema
        generation_config = {
            "temperature": 0.0, # Low temp for analytical tasks
            "response_mime_type": "application/json",
            "response_schema": AudioAnalysis
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )

        # 3. Create prompt
        final_prompt = AUDIO_ANALYSIS_PROMPT.format(question=question)

        # 4. Generate content (pass the file object AND the prompt)
        logger.info("Sending audio and prompt to Gemini...")
        response = model.generate_content([gemini_file, final_prompt])
        
        return json.loads(response.text)

    except Exception as e:
        logger.exception("Gemini Audio Analysis failed: %s", e)
        raise
    finally:
        # Cleanup: Delete the file from Gemini's server to save space/privacy
        if gemini_file:
            try:
                logger.info("Deleting file from Gemini cloud storage...")
                gemini_file.delete()
            except Exception:
                pass

# ---------- Main Exported Function ----------
def analyze_audio_features(video_input: t.Union[str, t.IO], question: str) -> AudioAnalysis:
    """
    Main entry point:
    1. Extracts audio from video input
    2. Uploads audio to Gemini
    3. Returns structured AudioAnalysis
    """
    audio_path = None
    try:
        # 1. Extract Audio
        audio_path = extract_audio_from_video(video_input, out_audio_ext=".mp3")
        
        # 2. Analyze with LLM
        llm_json = _call_gemini_with_audio(question=question, audio_path=audio_path)
        
        # 3. Validate and Return
        analysis = AudioAnalysis(**llm_json)
        return analysis

    finally:
        # Cleanup local audio file
        if (
            audio_path
            and os.path.exists(audio_path)
            and not _is_persistent_audio_asset(audio_path)
        ):
            try:
                os.remove(audio_path)
            except Exception:
                pass

# ---------- CLI Testing ----------
if __name__ == "__main__":
    import sys
    # Usage: python audio_analyse.py video.mp4 "Did you steal it?"
    if len(sys.argv) < 3:
        print('Usage: python audio_analyse.py <video_path> "<question>"')
        sys.exit(1)
        
    vid_path = sys.argv[1]
    q_text = sys.argv[2]
    
    try:
        result = analyze_audio_features(vid_path, q_text)
        print("\n--- Analysis Result ---")
        print(result.json(indent=2))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")