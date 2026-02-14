"""
Usage:
- As a module: from correctness import analyze_correctness; result = analyze_correctness(video_input, question)
  video_input can be:
    - a string path to a video file, e.g. "/path/to/video.mp4"
    - a file-like object with .read() and .name (e.g. streamlit uploaded file)
- As a standalone script for testing:
    python correctness.py /path/to/video.mp4 "Did you authorize the transaction?"

Requirements:
- ffmpeg must be installed on the system (moviepy relies on ffmpeg)
- openai whisper to be downloaded 
"""

import os
from dotenv import load_dotenv
import json
import tempfile
import typing as t
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from moviepy import VideoFileClip
import openai  # official Python client (older/newer variants compatible with client.audio.transcriptions.create pattern)
import logging
import re
import whisper


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Pydantic model ----------
class DeceptionSignal(str, Enum):
    LIE_DETECTED = "positive"
    NO_LIE_DETECTED = "negative"

class CorrectnessAnalysis(BaseModel):
    """Analysis result for textual correctness/deception detection."""
    signal: DeceptionSignal = Field(
        ...,
        description="Detection result: positive if lie detected, negative if not"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the detection decision"
    )

# ---------- System prompt ----------
CORRECTNESS_ANALYSIS_PROMPT = """You are an expert linguistic analyst specializing in deception detection through textual analysis. You will analyze a transcribed spoken answer to determine the likelihood of deception.

IMPORTANT CONTEXT:
- The text you're analyzing was transcribed from speech using an automated model (Whisper)
- There may be minor transcription errors, word substitutions, or missing punctuation
- Focus on content patterns rather than minor grammatical issues
- The speaker's native language proficiency is unknown

YOUR TASK:
Analyze the provided answer for linguistic and content-based indicators of deception or truthfulness.

ANALYZE FOR THESE DECEPTION INDICATORS:

1. RELEVANCE & DIRECTNESS:
   - Does the answer directly address the question asked?
   - Is there topic evasion or deflection to unrelated subjects?
   - Are there unnecessary tangents or diversions?

2. SPECIFICITY & DETAIL:
   - Is the answer vague or lacking concrete details?
   - Are there specific facts, names, numbers, or timeframes provided?
   - Is the level of detail appropriate (too little or suspiciously excessive)?

3. CONSISTENCY & LOGIC:
   - Are there internal contradictions within the answer?
   - Does the narrative flow logically and temporally?
   - Do different parts of the answer align with each other?

4. LINGUISTIC DECEPTION MARKERS:
   - Excessive hedging: "maybe", "I think", "kind of", "sort of", "probably"
   - Truth emphasis: "honestly", "to be honest", "frankly", "to tell you the truth", "believe me", "I swear"
   - Distancing language: referring to self in third person, passive voice overuse
   - Cognitive load signs: excessive repetition, self-corrections, incomplete thoughts
   - Overgeneralization: "always", "never", "everyone knows"

5. EVASIVENESS PATTERNS:
   - Answering a question with another question
   - Providing non-answer responses ("that's a good question", "it depends")
   - Attacking the question's validity instead of answering
   - Claiming memory problems excessively

6. EMOTIONAL INCONGRUENCE:
   - Does the emotional tone match the content?
   - Is there inappropriate certainty or uncertainty?
   - Are there defensive reactions or over-justifications?

7. ANSWER COMPLETENESS:
   - Is the answer complete or suspiciously brief?
   - Does it leave obvious gaps or unanswered aspects?



CRITICAL REMINDERS:
- Account for potential transcription errors when evaluating
- Consider that some hedging might be cultural or personality-based
- A truthful person might also be nervous or uncertain
- Base your assessment on multiple indicators, not single phrases
- Be nuanced - very few answers are purely 0.0 or 1.0

---
QUESTION: {question}

TRANSCRIBED ANSWER TO ANALYZE: {transcribed_text}

Provide your analysis:
"""

# ---------- Helpers ----------
def _is_path_like(obj) -> bool:
    return isinstance(obj, str)

def _save_filelike_to_tempfile(filelike, suffix=""):
    """
    Save a file-like object (with .read()) to a temporary file and return path.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        # streamlit's UploadedFile has .read(), returns bytes
        f.write(filelike.read())
    return tmp_path

def extract_audio_from_video(video_input: t.Union[str, t.IO], out_audio_ext: str = ".mp3") -> str:
    """
    Accepts a video path or file-like object. Returns path to audio file (mp3 by default).
    Requires ffmpeg (moviepy uses it).
    """
    needs_cleanup = []
    try:
        if _is_path_like(video_input):
            video_path = video_input
        else:
            # file-like: save to temp file
            suffix = ""
            name = getattr(video_input, "name", "")
            if name and "." in name:
                suffix = "." + name.split(".")[-1]
            video_path = _save_filelike_to_tempfile(video_input, suffix=suffix)
            needs_cleanup.append(video_path)

        logger.info("Loading video for audio extraction: %s", video_path)
        clip = VideoFileClip(video_path)
        tmp_audio_fd, tmp_audio_path = tempfile.mkstemp(suffix=out_audio_ext)
        os.close(tmp_audio_fd)
        # write_audiofile will call ffmpeg under the hood
        clip.audio.write_audiofile(tmp_audio_path, verbose=False, logger=None)
        clip.reader.close()
        clip.audio.reader.close_proc()
        return tmp_audio_path
    except Exception as e:
        logger.exception("Failed to extract audio: %s", e)
        raise
    finally:
        # If we created a temp video path from a file-like object, we keep the audio but remove temp video.
        for p in needs_cleanup:
            try:
                os.remove(p)
            except Exception:
                pass

def transcribe_with_openai_whisper(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio using a local Whisper model (no API).
    
    Parameters:
        audio_path: path to extracted audio file
        model_size: one of ["tiny", "base", "small", "medium", "large"]
    
    Returns:
        Transcribed text string.
    """
    try:
        logger.info("Loading local Whisper model: %s", model_size)
        model = whisper.load_model(model_size)

        logger.info("Transcribing audio: %s", audio_path)
        result = model.transcribe(audio_path)

        text = result.get("text", "").strip()

        if not text:
            logger.warning("Whisper returned empty transcription.")

        return text

    except Exception as e:
        logger.exception("Local Whisper transcription failed: %s", e)
        raise


def _call_llm_and_get_json(question: str, transcript: str, openai_model: str = "gpt-4o-mini") -> dict:
    """
    Call an LLM (OpenAI chat endpoint) with the system prompt and get back JSON.
    This function assumes OPENAI_API_KEY is set. It asks the model to return strict JSON matching the schema.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    openai.api_key = api_key

    system_prompt = CORRECTNESS_ANALYSIS_PROMPT.format(question=question, transcribed_text=transcript)

    # We instruct the model to return JSON only, matching our pydantic model:
    user_message = (
        "Return only JSON matching this shape: "
        '{"signal":"positive"|"negative","reasoning":"string"}\n\n'
        "Do not add commentary outside JSON. Keep reasoning concise."
    )

    # We'll use ChatCompletion if available
    try:
        resp = openai.ChatCompletion.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=800,
            temperature=0.0,
        )
        # different client versions have different shapes
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if choices:
            content = choices[0]["message"]["content"]
        else:
            # try attribute access
            content = getattr(resp, "choices")[0].message["content"]
    except Exception as e:
        logger.exception("ChatCompletion call failed: %s", e)
        raise

    # Extract JSON blob from model output (strip anything outside braces)
    json_text = None
    # First try direct parse
    try:
        json_text = content.strip()
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # heuristic: find first {...} block
        m = re.search(r"\{(?:[^{}]|(?R))*\}", content, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return parsed
            except Exception:
                logger.exception("Failed to parse JSON found in model output: %s", m.group(0))
        # If all fails, raise with model output for debugging
        raise ValueError(f"Could not parse JSON from model output. Raw output:\n{content}")

# ---------- Main exported function ----------
def analyze_correctness(video_input: t.Union[str, t.IO], question: str, cleanup_audio: bool = True) -> CorrectnessAnalysis:
    """
    Run the correctness module:
    - extracts audio from the provided video_input
    - transcribes audio via OpenAI Whisper
    - calls an LLM with CORRECTNESS_ANALYSIS_PROMPT and expects strict JSON exactly matching CorrectnessAnalysis

    Returns:
      CorrectnessAnalysis instance

    Side effects:
      prints the JSON result when called as a script; also logs.
    """
    audio_path = None
    try:
        audio_path = extract_audio_from_video(video_input, out_audio_ext=".mp3")
        logger.info("Audio extracted to %s", audio_path)
        transcription = transcribe_with_openai_whisper(audio_path)
        logger.info("Received transcription (truncated): %s", transcription[:200])

        llm_json = _call_llm_and_get_json(question=question, transcript=transcription)

        # Validate with pydantic
        try:
            analysis = CorrectnessAnalysis(**llm_json)
        except ValidationError as ve:
            # If validation fails, attempt to coerce or report errors
            logger.exception("Validation error when creating CorrectnessAnalysis from LLM output: %s", ve)
            raise

        return analysis
    finally:
        if cleanup_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


# ---------- CLI for standalone testing ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python correctness.py /path/to/video.mp4 \"Question text here\"")
        sys.exit(1)
    video_path = sys.argv[1]
    question_text = sys.argv[2]

    try:
        result = analyze_correctness(video_path, question_text)
        # Print as JSON to the console (for your final_decide module to read easily)
        print(result.json(indent=2))
    except Exception as e:
        logger.exception("Error running correctness analysis: %s", e)
        print("Error:", e)
        sys.exit(2)
