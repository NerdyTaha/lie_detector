import google.generativeai as genai

def _call_llm_and_get_json(question: str, transcript: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Call Gemini 2.5 Flash to analyze the transcript. 
    Uses native structured output (JSON mode) with the Pydantic schema.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    genai.configure(api_key=api_key)

    # Configure the model to enforce the Pydantic schema structure
    # We pass the schema class directly to response_schema
    generation_config = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": CorrectnessAnalysis
    }

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )

    # Format the existing global prompt template with the specific inputs
    final_prompt = CORRECTNESS_ANALYSIS_PROMPT.format(
        question=question, 
        transcribed_text=transcript
    )

    try:
        # Generate content
        response = model.generate_content(final_prompt)
        
        # Gemini handles the JSON serialization automatically via response_schema
        # We just need to parse the text back into a dict
        return json.loads(response.text)

    except Exception as e:
        logger.exception("Gemini API call failed: %s", e)
        raise