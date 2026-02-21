"""
visual_analyse.py

Creates a visual summary (collage) of a short video by sampling frames at even intervals,
arranging them into a single image, and sending that image to an LLM for behavioral analysis.

Usage:
- As module:
    from visual_analyse import analyze_visual
    result = analyze_visual("/path/to/video.mp4", question="Did you do X?", num_frames=8)
- Standalone (for testing):
    python visual_analyse.py /path/to/video.mp4 "Optional question text"

Requirements:
pip install opencv-python numpy pydantic google-generativeai pillow
ffmpeg is NOT required for this module (cv2.VideoCapture used).
"""

import os
import tempfile
import math
import json
import typing as t
import logging
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1) System Prompt
# ---------------------------------------------------------
VISUAL_ANALYSIS_PROMPT = """You are an expert in behavioral analysis and visual deception detection. You will analyze a collage of video frames showing a person answering a question to identify visual indicators of deception.

IMPORTANT CONTEXT:
- You are viewing a COLLAGE of frames sampled evenly across the video timeline.
- Frames are arranged in a grid from left-to-right, top-to-bottom (chronological order).
- Look for patterns and CHANGES across frames, not just isolated moments.
- The person knows they are being recorded (baseline nervousness is expected).

YOUR TASK:
Analyze the visual and behavioral indicators across the frames to determine likelihood of deception.

ANALYZE FOR THESE VISUAL DECEPTION INDICATORS:

1. EYE BEHAVIOR & GAZE PATTERNS:
   - Avoiding direct eye contact (looking away, down, or sideways)
   - Excessive blinking or rapid eye movements
   - Looking up-right (fabrication) vs. up-left (recall)
   - Staring too intensely (overcompensating)
   - IMPORTANT: Brief look-aways for thought are NORMAL; sustained avoidance is suspicious.

2. FACIAL EXPRESSIONS & MICROEXPRESSIONS:
   - Asymmetric facial expressions
   - Microexpressions that contradict stated emotion (fear, contempt, etc.)
   - Forced or fake smiles (mouth smiles but eyes don't)
   - Expressions not matching the content (smiling when serious)
   - Tension in jaw, lips pressed together, or mouth covering

3. SELF-TOUCH & ADAPTORS (Self-Soothing):
   - Touching face, especially nose, mouth, or ears
   - Rubbing neck or scratching
   - Hand-to-face gestures (hiding mouth)
   - Playing with hair, jewelry, or clothing
   - IMPORTANT: Repetitive self-touch during key claims = STRESS INDICATOR.

4. HAND & ARM MOVEMENTS:
   - Sudden stillness or "freezing" (inhibited gestures)
   - Hands hidden (in pockets, behind back)
   - Defensive barriers (crossed arms)
   - Grooming behaviors
   - Lack of illustrators (natural hand gestures)

5. BODY POSTURE & POSITIONING:
   - Leaning away from camera (distancing)
   - Closed posture (hunched, crossed limbs)
   - Shifting or fidgeting in seat
   - Rigid, frozen posture

6. CONGRUENCE & CONSISTENCY:
   - Do facial expressions align with expected emotions?
   - Does body language match the confidence of the answer?
   - Are there contradictions (saying "yes" while shaking head)?

CRITICAL DECISION FRAMEWORK:

✓ NORMAL BEHAVIORS (Not Deceptive):
  - Brief look-aways to think
  - Natural hand gestures accompanying speech
  - Slight nervousness throughout (camera anxiety)
  - Occasional adjustments of posture

✗ SUSPICIOUS PATTERNS (Potentially Deceptive):
  - CLUSTERS of indicators appearing together
  - SUDDEN CHANGES in behavior at specific moments
  - Behavior that contradicts verbal content
  - Self-soothing behaviors concentrated around key claims
  - Overcontrolled or frozen expressions

DECISION CRITERIA:
- POSITIVE (Lie Detected): Multiple clear indicators from different categories, significant behavioral changes, strong incongruence.
- NEGATIVE (No Lie Detected): Natural body language variation, behaviors consistent with normal recording anxiety, genuine emotional expressions.

---
QUESTION BEING ANSWERED: {question}

Analyze the collage and provide your assessment in the required format.
"""

# ---------------------------------------------------------
# 2) Pydantic Models
# ---------------------------------------------------------
class DeceptionSignal(str, Enum):
    LIE_DETECTED = "positive"
    NO_LIE_DETECTED = "negative"

class VisualAnalysisResult(BaseModel):
    """Analysis result for visual/behavioral deception detection."""
    
    signal: DeceptionSignal = Field(
        ...,
        description="Detection result: positive if lie detected, negative if not"
    )
    
    reasoning: str = Field(
        ...,
        description="Explanation for the visual-based detection decision"
    )

# ---------------------------------------------------------
# 3) LLM Implementation
# ---------------------------------------------------------
def call_llm_with_image(image_path: str, prompt: str) -> dict:
    """
    Calls Gemini 2.5 Flash with the collage image and the analysis prompt.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    genai.configure(api_key=api_key)

    try:
        # Load image using Pillow
        img = Image.open(image_path)

        # Configure model to return JSON matching the Pydantic schema
        generation_config = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": VisualAnalysisResult
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )

        logger.info("Sending collage and prompt to Gemini...")
        
        # Pass both the text prompt and the PIL image object
        response = model.generate_content([prompt, img])
        
        return json.loads(response.text)

    except Exception as e:
        logger.exception("Gemini Visual Analysis failed: %s", e)
        raise

# ---------------------------------------------------------
# 4) Helpers: Video Processing & Collage
# ---------------------------------------------------------

def _is_path_like(obj) -> bool:
    return isinstance(obj, str)

def _save_filelike_to_tempfile(filelike, suffix=""):
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(filelike.read())
    return tmp_path

def _to_send_collage_path(suffix: str = ".jpg") -> str:
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    return os.path.join(ASSETS_DIR, f"to_send_collage{ext}")

def _is_persistent_collage_asset(path: str) -> bool:
    abs_path = os.path.abspath(path)
    assets_path = os.path.abspath(ASSETS_DIR)
    return abs_path.startswith(assets_path + os.sep) and os.path.basename(abs_path).startswith("to_send_collage")

def extract_evenly_spaced_frames(video_path: str, num_frames: int = 8) -> t.List[np.ndarray]:
    """
    Extract `num_frames` evenly spaced frames from video using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # If video metadata is broken or empty, try to read a few frames manually
    if total_frames <= 0:
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise RuntimeError("No frames extracted from video.")
            
        # Sample evenly from the list we read
        indices = np.linspace(0, len(frames)-1, min(num_frames, len(frames))).astype(int)
        return [frames[i] for i in indices]

    # Normal efficient seek-based extraction
    num_to_sample = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, num_to_sample).astype(int)

    sampled_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Frame read failed at index %s; skipping.", idx)
            continue
        sampled_frames.append(frame)

    cap.release()

    if not sampled_frames:
        raise RuntimeError("Failed to extract any frames from the video.")

    logger.info("Extracted %d frames.", len(sampled_frames))
    return sampled_frames

def make_collage(frames: t.List[np.ndarray], thumb_size: t.Tuple[int, int] = (320, 180), padding: int = 5) -> np.ndarray:
    """
    Build a grid collage from a list of frames.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to build a collage.")

    # Calculate grid dimensions
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    w, h = thumb_size
    canvas_w = grid_cols * w + padding * (grid_cols + 1)
    canvas_h = grid_rows * h + padding * (grid_rows + 1)

    # Create white canvas
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255 

    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= n:
                break
            frame = frames[idx]
            # Resize frame to thumbnail size
            thumb = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            
            x = padding + c * (w + padding)
            y = padding + r * (h + padding)
            canvas[y:y+h, x:x+w] = thumb
            idx += 1

    return canvas

def save_image_bgr_to_tempfile(img_bgr: np.ndarray, suffix=".jpg", jpeg_quality: int = 85) -> str:
    output_path = _to_send_collage_path(suffix=suffix)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success = cv2.imwrite(output_path, img_bgr, encode_params)
    if not success:
        raise RuntimeError("Failed to write collage image to disk.")
    return output_path

# ---------------------------------------------------------
# 5) Main Exported Function
# ---------------------------------------------------------
def build_visual_collage(video_input: t.Union[str, t.IO],
                         num_frames: int = 8,
                         thumb_size: t.Tuple[int, int] = (320, 180),
                         cleanup_video_temp: bool = True) -> str:
    """
    Creates and saves a collage image from video_input.
    Returns path to the saved jpg.
    """
    temp_video_path = None
    created_temp_video = False
    try:
        if _is_path_like(video_input):
            temp_video_path = video_input
        else:
            temp_video_path = _save_filelike_to_tempfile(video_input, suffix=".mp4")
            created_temp_video = True

        frames = extract_evenly_spaced_frames(temp_video_path, num_frames=num_frames)
        collage_bgr = make_collage(frames, thumb_size=thumb_size)
        collage_path = save_image_bgr_to_tempfile(collage_bgr, suffix=".jpg")
        
        return collage_path
    finally:
        if created_temp_video and cleanup_video_temp and temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

def analyze_visual(video_input: t.Union[str, t.IO],
                   question: t.Optional[str] = None,
                   num_frames: int = 8,
                   thumb_size: t.Tuple[int, int] = (320, 180),
                   cleanup_collage: bool = True) -> VisualAnalysisResult:
    """
    Main entry point for visual analysis.
    1. Extracts frames and builds a collage.
    2. Sends collage + prompt to Gemini.
    3. Returns structured VisualAnalysisResult.
    """
    collage_path = None
    try:
        # 1. Build Collage
        collage_path = build_visual_collage(video_input, num_frames=num_frames, thumb_size=thumb_size)
        logger.info("Collage built at: %s", collage_path)

        # 2. Format Prompt
        # We use safe formatting in case question is None
        q_text = question if question else "No specific question context provided."
        final_prompt = VISUAL_ANALYSIS_PROMPT.format(question=q_text)

        # 3. Call LLM
        llm_resp = call_llm_with_image(collage_path, final_prompt)
        
        # 4. Validate and Return
        result = VisualAnalysisResult(**llm_resp)
        return result

    finally:
        # Cleanup collage file
        if (
            cleanup_collage
            and collage_path
            and os.path.exists(collage_path)
            and not _is_persistent_collage_asset(collage_path)
        ):
            try:
                os.remove(collage_path)
            except Exception:
                pass

# ---------------------------------------------------------
# CLI Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visual_analyse.py /path/to/video.mp4 \"Optional question text\"")
        sys.exit(1)

    video_path = sys.argv[1]
    question_text = sys.argv[2] if len(sys.argv) >= 3 else "Unknown Question"

    try:
        print(f"Analyzing video: {video_path}")
        result = analyze_visual(video_path, question=question_text, cleanup_collage=False)
        print("\n--- Visual Analysis Result ---")
        print(result.json(indent=2))
    except Exception as e:
        logger.exception("Error running visual analysis")
        print("Error:", e)
        sys.exit(2)