"""
visual_analyse.py

Creates a visual summary (collage) of a short video by sampling frames at even intervals,
arranging them into a single image, and (optionally) sending that image to an LLM.

THREE BLANKS YOU MUST FILL:
- VISUAL_ANALYSIS_PROMPT: the prompt string for the LLM
- VisualAnalysisResult: a pydantic model describing the LLM response
- call_llm_with_image(image_path: str, prompt: str) -> dict: the function that calls your LLM/image API

Usage:
- As module:
    from visual_analyse import analyze_visual
    result = analyze_visual("/path/to/video.mp4", question="Did you do X?", num_frames=8)
- Standalone (for testing):
    python visual_analyse.py /path/to/video.mp4 "Optional question text"

Requirements:
pip install opencv-python numpy pydantic
ffmpeg is NOT required for this module (cv2.VideoCapture used). If that fails on some containers,
you can switch to moviepy-based extraction.
"""

import os
import tempfile
import math
import typing as t
import logging
import numpy as np
import cv2
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# 1) Prompt 
VISUAL_ANALYSIS_PROMPT = """You are an expert in behavioral analysis and visual deception detection. You will analyze a collage of video frames showing a person answering a question to identify visual indicators of deception.

IMPORTANT CONTEXT:
- You are viewing a COLLAGE of frames sampled evenly across the video timeline
- Frames are arranged in a grid from left-to-right, top-to-bottom (chronological order)
- Look for patterns and CHANGES across frames, not just isolated moments
- The person knows they are being recorded (baseline nervousness is expected)

YOUR TASK:
Analyze the visual and behavioral indicators across the frames to determine likelihood of deception.

ANALYZE FOR THESE VISUAL DECEPTION INDICATORS:

1. EYE BEHAVIOR & GAZE PATTERNS:
   - Avoiding direct eye contact (looking away, down, or sideways)
   - Excessive blinking or rapid eye movements
   - Looking up-right (often associated with fabrication) vs. up-left (recall)
   - Staring too intensely (overcompensating to appear truthful)
   - Eye direction changes when discussing specific details
   - IMPORTANT: Brief look-aways for thought are NORMAL; sustained avoidance is suspicious

2. FACIAL EXPRESSIONS & MICROEXPRESSIONS:
   - Asymmetric facial expressions (one side different from other)
   - Microexpressions that contradict stated emotion (flash of fear, contempt, etc.)
   - Forced or fake smiles (mouth smiles but eyes don't - no crow's feet)
   - Expressions not matching the content (smiling when serious, serious when casual)
   - Timing: delayed emotional reactions or emotions that fade too quickly
   - Tension in jaw, lips pressed together, or mouth covering

3. SELF-TOUCH & ADAPTORS (Self-Soothing Behaviors):
   - Touching face, especially nose, mouth, or ears
   - Rubbing neck or scratching
   - Hand-to-face gestures (hiding mouth, touching nose)
   - Playing with hair, jewelry, or clothing
   - IMPORTANT: Distinguish between:
     * Occasional adjustments = NORMAL
     * Repetitive self-touch during key claims = STRESS INDICATOR

4. HAND & ARM MOVEMENTS:
   - Sudden stillness or "freezing" (inhibited gestures)
   - Hands hidden (in pockets, behind back, under table)
   - Defensive barriers (crossed arms, holding objects as shields)
   - Grooming behaviors (fixing clothes, adjusting appearance)
   - Unnatural hand positions or awkward gestures
   - Lack of illustrators (hand gestures that accompany speech)

5. BODY POSTURE & POSITIONING:
   - Leaning away from camera (distancing)
   - Closed posture (hunched shoulders, crossed limbs)
   - Shifting or fidgeting in seat
   - Creating physical barriers between self and camera
   - Sudden posture changes when specific topics arise
   - Rigid, frozen posture (over-controlling body)

6. HEAD MOVEMENTS:
   - Excessive nodding or head shaking
   - Head movements that contradict verbal statements (nodding while saying "no")
   - Tilting head down (submission/shame)
   - Looking away during critical moments

7. SIGNS OF PHYSIOLOGICAL STRESS:
   - Visible sweating (forehead, upper lip)
   - Flushed face or color changes
   - Swallowing or throat clearing
   - Visible tension (clenched jaw, tense shoulders)
   - Rapid breathing (chest movement)

8. TEMPORAL PATTERNS ACROSS FRAMES:
   - Look for CHANGES in behavior across the timeline
   - Does body language shift when answering vs. when question was asked?
   - Increased tension/movement in middle frames (fabricating) vs. start/end
   - Relaxation after difficult question suggests relief (deception completed)

9. CONGRUENCE & CONSISTENCY:
   - Do facial expressions align with expected emotions?
   - Does body language match the confidence of the answer?
   - Are there contradictions (saying "yes" while shaking head)?

CRITICAL DECISION FRAMEWORK:

✓ NORMAL BEHAVIORS (Not Deceptive):
  - Brief look-aways to think
  - Natural hand gestures accompanying speech
  - Slight nervousness throughout (camera anxiety)
  - Occasional adjustments of posture or clothing
  - Cultural differences in eye contact norms

✗ SUSPICIOUS PATTERNS (Potentially Deceptive):
  - CLUSTERS of indicators appearing together
  - SUDDEN CHANGES in behavior at specific moments
  - Behavior that contradicts verbal content
  - Self-soothing behaviors concentrated around key claims
  - Overcontrolled or frozen expressions
  - Multiple indicators from different categories

✓ LOOK FOR PATTERNS, NOT SINGLE INSTANCES:
  - One hand-to-face touch ≠ deception
  - Multiple self-touch gestures + gaze aversion + expression mismatch = SUSPICIOUS
  - Changes from baseline (early frames) to later frames matter more than absolute behaviors

COLLAGE ANALYSIS STRATEGY:
1. Observe the first frame(s) as potential baseline behavior
2. Track changes across the frame sequence (left-to-right, top-to-bottom)
3. Note if suspicious behaviors cluster in specific frames (likely during key claims)
4. Consider overall pattern: natural variation vs. stress markers

DECISION CRITERIA:
- POSITIVE (Lie Detected): Multiple clear indicators from different categories, significant behavioral changes during what appears to be critical moments, strong incongruence between expression and expected emotion
- NEGATIVE (No Lie Detected): Natural body language variation, behaviors consistent with normal recording anxiety, genuine emotional expressions, absence of clustered stress indicators

---
QUESTION BEING ANSWERED: {question}

Analyze the collage and provide your assessment in the required format."""

# 2) Pydantic model(s) 
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

# 3) LLM call: a stub 
def call_llm_with_image(image_path: str, prompt: str) -> dict:
    """
    TODO: implement the call to your LLM/image+text endpoint here.

    Should accept:
      - image_path: path to the collage image
      - prompt: the prompt/context you want to send (you can format VISUAL_ANALYSIS_PROMPT with question/metadata)

    Expected return: a dict compatible with your VisualAnalysisResult when validated.

    For now, raise NotImplementedError so users know this is pending.
    """
    raise NotImplementedError("Implement call_llm_with_image(image_path, prompt) to call your LLM/API.")


# ----------------------------
# Helpers: I/O & collage logic
# ----------------------------

def _is_path_like(obj) -> bool:
    return isinstance(obj, str)

def _save_filelike_to_tempfile(filelike, suffix=""):
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(filelike.read())
    return tmp_path

def _ensure_video_path(video_input: t.Union[str, t.IO]) -> str:
    """
    Accept either a path or a file-like object (e.g. streamlit uploaded file).
    Returns a filesystem path to the video.
    Caller is responsible for removing returned temp file if necessary.
    """
    if _is_path_like(video_input):
        return video_input
    else:
        name = getattr(video_input, "name", "")
        suffix = ""
        if name and "." in name:
            suffix = "." + name.split(".")[-1]
        return _save_filelike_to_tempfile(video_input, suffix=suffix)

def extract_evenly_spaced_frames(video_path: str, num_frames: int = 8) -> t.List[np.ndarray]:
    """
    Use cv2.VideoCapture to read the video and extract `num_frames` evenly spaced frames.
    Returns list of BGR images (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = (total_frames / fps) if fps > 0 else 0

    logger.info("Video opened: %s | frames=%s fps=%.2f duration=%.2fs",
                video_path, total_frames, fps or 0.0, duration)

    if total_frames <= 0:
        # Fallback: try reading frames sequentially until EOF and collect them (rare)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # sample from collected frames
        if not frames:
            raise RuntimeError("No frames extracted from video.")
        indices = np.linspace(0, len(frames)-1, min(num_frames, len(frames))).astype(int)
        sampled = [frames[i] for i in indices]
        return sampled

    # determine frame indices to sample
    num_to_sample = min(num_frames, total_frames)
    # evenly space across the whole video (including first and last)
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
    - frames: list of BGR numpy arrays
    - thumb_size: size (w, h) to resize each thumbnail to
    Returns the collage as a BGR numpy array.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to build a collage.")

    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    w, h = thumb_size
    canvas_w = grid_cols * w + padding * (grid_cols + 1)
    canvas_h = grid_rows * h + padding * (grid_rows + 1)

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white background

    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= n:
                break
            frame = frames[idx]
            # resize while maintaining aspect in case shape differs
            thumb = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            x = padding + c * (w + padding)
            y = padding + r * (h + padding)
            canvas[y:y+h, x:x+w] = thumb
            idx += 1

    return canvas

def save_image_bgr_to_tempfile(img_bgr: np.ndarray, suffix=".jpg", jpeg_quality: int = 85) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # use cv2.imwrite with JPEG quality
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success = cv2.imwrite(tmp_path, img_bgr, encode_params)
    if not success:
        raise RuntimeError("Failed to write collage image to disk.")
    return tmp_path

# ----------------------------
# High-level API
# ----------------------------

def build_visual_collage(video_input: t.Union[str, t.IO],
                         num_frames: int = 8,
                         thumb_size: t.Tuple[int, int] = (320, 180),
                         cleanup_video_temp: bool = True) -> str:
    """
    Creates and saves a collage image from video_input.
    Returns the path to the saved collage image.

    If `video_input` is a file-like object, a temporary file will be created and removed (if cleanup_video_temp True).
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
        logger.info("Collage saved to %s", collage_path)
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
    High-level analyse function. Steps:
    1) build collage image
    2) format prompt (you can include the question and any metadata)
    3) call LLM with the image and prompt (call_llm_with_image) -> returns dict
    4) validate dict into VisualAnalysisResult (pydantic) and return

    NOTE: call_llm_with_image is intentionally left as NotImplemented. Replace it with your API call.
    """
    collage_path = build_visual_collage(video_input, num_frames=num_frames, thumb_size=thumb_size)
    logger.info("Collage built at: %s", collage_path)

    # Prepare prompt - you can format the placeholder prompt with question and metadata:
    prompt = VISUAL_ANALYSIS_PROMPT
    if question:
        # Many prompts append question/context at the end:
        prompt = (prompt + "\n\nQUESTION: {q}\n").format(q=question)

    # Call LLM (stubbed)
    llm_resp = call_llm_with_image(collage_path, prompt)  # <-- implement this

    # Validate into pydantic model - your VisualAnalysisResult should be defined by you
    result = VisualAnalysisResult(**llm_resp)  # <-- you will replace stub model with real one

    # Optionally cleanup collage file
    if cleanup_collage:
        try:
            os.remove(collage_path)
        except Exception:
            pass

    return result

# ----------------------------
# CLI for quick testing
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visual_analyse.py /path/to/video.mp4 \"Optional question text\"")
        sys.exit(1)

    video_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) >= 3 else None

    try:
        collage_path = build_visual_collage(video_path, num_frames=8)
        print(f"Collage written to: {collage_path}")

        # Try to call LLM only if implemented
        try:
            prompt = VISUAL_ANALYSIS_PROMPT
            if question:
                prompt = (prompt + "\n\nQUESTION: {q}\n").format(q=question)
            llm_out = call_llm_with_image(collage_path, prompt)
            print("LLM response (raw):")
            print(llm_out)
            # If you want to validate with the filled pydantic model, do:
            # result = VisualAnalysisResult(**llm_out)
            # print(result.json(indent=2))
        except NotImplementedError:
            print("call_llm_with_image is not implemented - collage created but LLM call is stubbed.")
    except Exception as e:
        logger.exception("Error running visual analysis: %s", e)
        print("Error:", e)
        sys.exit(2)
