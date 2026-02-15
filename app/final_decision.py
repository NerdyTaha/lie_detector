"""
final_decision.py

Aggregates the analysis results from Correctness (Text), Audio, and Visual modules.
Sends the combined data to Gemini 2.5 Flash to generate a final verdict and synthesis.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import google.generativeai as genai

# Import return types from other modules for type hinting (optional but good practice)
# Assuming the other files are in the same directory
from correctness import CorrectnessAnalysis
from audio_analyse import AudioAnalysis
from visual_analyse import VisualAnalysisResult

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Pydantic Models ----------
class FinalSignal(str, Enum):
    HIGH_LIKELIHOOD_LIE = "LIE"
    POSSIBLE_LIE = "POSSIBLE_LIE"
    INCONCLUSIVE = "INCONCLUSIVE"
    TRUTHFUL = "TRUTHFUL"

class FinalVerdict(BaseModel):
    """The final synthesized decision based on multi-modal analysis."""
    
    verdict: FinalSignal = Field(
        ...,
        description="The final classification of the subject's response."
    )
    
    confidence_score: int = Field(
        ...,
        description="Confidence level in the verdict from 0 to 100."
    )
    
    synthesis_reasoning: str = Field(
        ...,
        description="A comprehensive summary explaining how the three signals (Text, Audio, Visual) were weighed to reach this conclusion. Highlight incongruences if any."
    )

# ---------- System Prompt ----------
FINAL_DECISION_PROMPT = """You are the Lead Forensic Investigator for a lie detection system.
You have received three expert reports analyzing a subject's response to an interrogation question.

YOUR INPUTS:
1. QUESTION: "{question}"
2. TEXTUAL/CONTENT ANALYSIS: Checks for logical inconsistencies, evasion, and linguistic markers.
3. AUDIO ANALYSIS: Checks for vocal tremors, pitch changes, and hesitation markers.
4. VISUAL ANALYSIS: Checks for micro-expressions, gaze avoidance, and body language.

INPUT DATA:
---
[REPORT 1: TEXTUAL ANALYSIS]
{text_report}
---
[REPORT 2: AUDIO ANALYSIS]
{audio_report}
---
[REPORT 3: VISUAL ANALYSIS]
{visual_report}
---

YOUR TASK:
Synthesize these three reports into a Final Verdict.

GUIDELINES FOR SYNTHESIS:
1. LOOK FOR INCONGRUENCE: This is the strongest sign of deception. (e.g., The person sounds confident [Audio: Negative] but their body language is defensive [Visual: Positive]).
2. WEIGHTING: 
   - Visual and Audio cues are often more reliable for emotional leakage than Text.
   - Text is more reliable for logical inconsistencies.
3. FALSE POSITIVES: If the subject is nervous (Audio/Visual) but the story is perfectly consistent (Text), consider "Truthful but Nervous" rather than "Lie".
4. CLUSTERS: A "Lie" verdict requires a cluster of deceptive indicators across at least two modalities, or overwhelming evidence in one.

Determine the final verdict (LIE, POSSIBLE_LIE, INCONCLUSIVE, or TRUTHFUL) and provide a confidence score.
"""

def generate_final_verdict(
    question: str, 
    text_result: CorrectnessAnalysis, 
    audio_result: AudioAnalysis, 
    visual_result: VisualAnalysisResult
) -> FinalVerdict:
    """
    Calls Gemini to synthesize the three analysis results into a final decision.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    genai.configure(api_key=api_key)

    try:
        # Convert pydantic models to JSON strings for the prompt
        text_json = text_result.json()
        audio_json = audio_result.json()
        visual_json = visual_result.json()

        # Format prompt
        final_prompt = FINAL_DECISION_PROMPT.format(
            question=question,
            text_report=text_json,
            audio_report=audio_json,
            visual_report=visual_json
        )

        generation_config = {
            "temperature": 0.1, # Low temperature for strict synthesis
            "response_mime_type": "application/json",
            "response_schema": FinalVerdict
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )

        logger.info("Sending aggregated reports to Gemini for final verdict...")
        response = model.generate_content(final_prompt)
        
        return FinalVerdict(**json.loads(response.text))

    except Exception as e:
        logger.exception("Final Decision generation failed: %s", e)
        raise

# CLI Testing
if __name__ == "__main__":
    # Mock data for testing
    print("This module is intended to be imported by app.py")