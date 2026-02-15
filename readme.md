# LLM powered lie detector 
- Leverages Gemini to detect lies. 

# how to run the app? 
- set up python venv with libraries: streamlit, fastapi, pydantic
- set up an .env file in your repo with api keys (make sure the var names are same as the ones in code)
- Execute the command for running the app:   streamlit run app/app.py 

## Engineering of the app
---
Problem statement/App description: 
A LLM powered application that inspects the video answer for an interrogation question and detects whether the person answering is lying or not.
---
- There would be an input text box as question. 
- Plus there would be a video input as question. (0-30 seconds video - since app is just mvp)
### UI 
app.py contains streamlit ui. It's connected to backend engine. 
### Backend engine: 
##### These backend modules can be tested stand-alone as well.
##### Correctness module 
    - This will take a video, extract audio from it, then convert it to text using OpenAI whisper.
    - The text will be sent to LLM and LLM will be tasked to identify relevance, correctness, etc, and give a signal as negative or positive (with context of input provided)
    - Positive if a possible lie detected. Negative if text does is likely not a lie.
    - Also, a reason for the signal. 
##### Audio_analyse module 
    - This will take the video and extract audio from it. 
    - The audio (with context of its extraction) will be sent to an LLM for processing. 
    - The LLM will be asked to look for patterns that are common in "lying" like stuttering and many filler words or pausing too much.
    - The LLM will then output a signal and reason for that signal. 
    positive signal if "lie" detected and negative signal if speech is truthful.
##### Visual_analyse module 
    - This will take a video, extract images from it (frames of video)
    - Then the images will be clubbed to make a collage - and it will be passed to LLM. 
    - The LLM will look at the images and will be prompted to identify patterns of lying.
    - It will give a signal (positive if lie detected & negative if lie not detected) and a reason for that signal. 

##### Final_decide module -> this would be done in app.py itslef.
    - Read signals and reason of the first 3 modules. 
    - aggregate them and provide a final decision of lie or not along with a reason.