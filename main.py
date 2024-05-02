from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse
import json
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    FunctionDeclaration,
    GenerativeModel,
    Tool,
)
from dotenv import load_dotenv
import os
import requests
import json


# Load environment variables from .env file
# Accessing gimini model from Vertex AI
load_dotenv()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

app = FastAPI()

# Initializing Vertex AI with your project ID and location
PROJECT_ID = "podchecks"
REGION = "europe-west2"
vertexai.init(project=PROJECT_ID, location=REGION)
MODEL_ID = "gemini-1.5-pro-preview-0409"
model = GenerativeModel(MODEL_ID)

def seconds_to_hms(seconds):
    """Converts seconds to a hours:minutes:seconds format without milliseconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)  # Convert seconds to an integer to avoid decimals
    return f"{hours:02}:{minutes:02}:{seconds:02}"

class YouTubeURL(BaseModel):
    url: str

class Claim(BaseModel):
    claim: str

def get_transcript_with_timestamps(youtube_url):
    """Fetches the transcript of a YouTube video from a given URL and returns text with timestamps."""
    try:
        # Parsing the URL to extract the video ID
        url_data = urllib.parse.urlparse(youtube_url)
        #print(url_data)
        query_params = urllib.parse.parse_qs(url_data.query)
        video_id = query_params["v"][0]  # Extracting the value of the 'v' parameter which is the video ID
        print(video_id)

        # Fetching the transcript using the extracted video ID
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Creating a formatted string with timestamps and text
        formatted_transcript = " ".join(f"[{seconds_to_hms(entry['start'])}-{seconds_to_hms(entry['start'] + entry['duration'])}] {entry['text']}" for entry in transcript)
        #formatted_transcript = "\n".join(f"[{entry['start']:.2f}-{entry['start'] + entry['duration']:.2f}] {entry['text']}" for entry in transcript)
        return formatted_transcript
    except Exception as e:
        print(f"Failed to fetch transcript for video URL {youtube_url}: {e}")
        return None


@app.post("/extract_claims/")
async def extract_claims(item: YouTubeURL):
    # Assuming your existing functions are imported and your model is set up correctly
    transcript_paragraph = get_transcript_with_timestamps(item.url)
    if not transcript_paragraph:
        raise HTTPException(status_code=404, detail="Transcript not found or could not be retrieved.")

    # You would need to adjust the following to actually call your model and parse the output
    prompt_text="""
    You are a podcast claim extractor analyzing a YouTube transcript. This transcript includes segments of speech, each accompanied by timestamps indicating its duration.

    Your task is to Extract all claims made during the discussion. For each claim, provide the timestamps in a format of hours, minutes, and seconds (HH:MM:SS), omitting fractions of seconds. Format your output as Dict where each key is labeled as 'claim' followed by a sequence number, and each value combines the timestamp and claim text in a clear and concise manner.

    Note: i have a next pipeline running on this dict iterating each claim to identify the fact of claims so, don't ommit dict format

    Example of expected output:
    {
    "claim1": "[00:00:04-00:00:03] Intermittent fasting is going to kill.",
    "claim2": "[00:00:27-00:00:30] A new study showed that intermittent fasting could increase your risk of death."
    }
      """
    
    
    try:

      generation_config = GenerationConfig(
      temperature=0.9,
      top_p=1.0,
      top_k=32,
      candidate_count=1,
      max_output_tokens=8192,
      )

      contents = [prompt_text + transcript_paragraph]
  
      response = model.generate_content(
      contents,
      generation_config=generation_config)

      clean_json_str = response.text.strip().strip('`').replace('\n', '')[4:]
      claims_dict = json.loads(clean_json_str)
      print(type(claims_dict))
      return claims_dict
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse the model output.")

#-----------------------*************************
# --------------------  Below is the logic for fact check endpoint

google_api_key = ''
PROJECT_ID = "podchecks"
REGION = "europe-west2"
vertexai.init(project=PROJECT_ID, location=REGION)

# Set model to Gemini 1.5 Pro
MODEL_ID = "gemini-1.5-pro-preview-0409"

# Set other api keys
wolfram_app_id = ''

# Simplifies the response given by Google Fact Check API
def simplify_fact_check_response(api_response):
  if not isinstance(api_response, dict) or 'claims' not in api_response:
    raise ValueError("Invalid API response format")

  simplified_response = []
  for claim in api_response.get('claims', []):
    if not isinstance(claim, dict):
        continue  # Skip improperly formatted claims
    claim_text = claim.get('text', 'No claim text provided')
    for review in claim.get('claimReview', []):
        if not isinstance(review, dict):
            continue  # Skip improperly formatted reviews
        simplified_review = {
            'claim_text': claim_text,
            'claimant': claim.get('claimant', 'Unknown'),
            'claim_date': claim.get('claimDate', 'No date provided'),
            'review_publisher': review.get('publisher', {}).get('name', 'Unknown publisher'),
            'review_url': review.get('url', 'No URL provided'),
            'review_title': review.get('title', 'No title provided'),
            'review_date': review.get('reviewDate', 'No date provided'),
            'textual_rating': review.get('textualRating', 'No rating provided'),
            'language': review.get('languageCode', 'en')  # Consider validation or dynamic handling
        }
        simplified_response.append(simplified_review)
  return simplified_response


def google_fact_check(query):
    print('Google fact check function called')
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {'query': query, 'key': google_api_key}
    response = requests.get(url, params=params)
    print(response)
    print(response.json())

    if not response.ok or 'claims' not in response.json() or len(response.json()['claims']) == 0:
        print("Fallback to Gemini model because of insufficient data from API.")
        return fallback_gemini_fact_check(query)  # Assuming fallback function exists
    else:
        simplified_response = simplify_fact_check_response(response.json())
        print(simplified_response)
        return simplified_response

def fallback_gemini_fact_check(query):
  # Use your local Gemini model to check the fact
  # Placeholder: Implement according to your Gemini model invocation
  # This should return a JSON formatted string
  prompt_text="""
  "You are a podcast fact checker.",
    "Your mission is to check the factual validity of claim for the provided claim.",
    "Format your response as a JSON object with the following structure:",
    "  {",
    "    \"claim\": \"[State the claim here]\",",
    "    \"analysis\": \"[Detailed analysis based on the sources consulted]\",",
    "    \"sources\": [\"[URL of the first source]\", \"[URL of the second source, etc.]\"]",
    "  }", 

  """
  # Set contents to send to the model
  contents = [prompt_text + query]

  # Counts tokens
  print(model.count_tokens(contents))

  generation_config = GenerationConfig(
      temperature=0.9,
      top_p=1.0,
      top_k=32,
      candidate_count=1,
      max_output_tokens=8192,
  )

  # Prompt the model to generate content
  response = model.generate_content(
      contents,
      generation_config=generation_config,

  )
    # Assume model_infer is the function that invokes the Gemini model
  return response.text


# Define function for using Wolfram Alpha
def wolfram_alpha(query):
  url = f"https://www.wolframalpha.com/api/v1/llm-api?input={query}&appid={wolfram_app_id}"
  response = requests.get(url)
  return response.text

# Define Vertex AI function schema for Google Fact Check
google_fact_check_func = FunctionDeclaration(
  name="google_fact_check",
  description="Use the Google Fact Check API to check if a claim has previously been fact checked as true or false. If it is found to be false this is one of the strongest indicators that it is not true. However, this relies on a claim as previously being false, if no evidence is found you should check other sources.",
  parameters={
      "type": "object",
      "properties": {"query": {"type": "string", "description": "The claim to check."}},
  },
)

# Define Vertex AI function schema for Wolfram Alpha
wolfram_alpha_func = FunctionDeclaration(
  name="wolfram_alpha",
  description="Use Wolfram Alpha to answer validate claims made of a scientific nature. WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc. Convert inputs to simplified keyword queries whenever possible (e.g. convert how many people live in France to France population)",
  parameters={
      "type": "object",
      "properties": {"query": {"type": "string", "description": "The claim to check."}},
  },
)

@app.post("/claim_factcheck")
async def extract_claims(item: Claim):
  # Assuming your existing functions are imported and your model is set up correctly

  # Create a Gemini tool with the declared functions
  fact_check_tool = Tool(
  function_declarations=[
    google_fact_check_func,
    #wolfram_alpha_func,
  ],
  )
  # Create the Gemini model with the defined tools
  model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a podcast fact checker.",
      "Your mission is to check the factual validity of claims using the provided tools.",
      "Format your response as a JSON object with the following structure:",
      "  {",
      "    \"claim\": \"[State the claim here]\",",
      "    \"analysis\": \"[Detailed analysis based on the sources consulted]\",",
      "    \"sources\": [\"[URL of the first source]\", \"[URL of the second source, etc.]\"]",
          " \"outcome\": [\"[Based on the analysis please assign outcome as True or False or Unsure]",
      "  }", 
    ],
    tools=[fact_check_tool], 
  )
  # Start the thread
  chat = model.start_chat()
  
  response = chat.send_message(item.claim)
  response = response.candidates[0].content.parts[0]
  #print(json.loads(response.text))
  clean_json_str = response.text.strip().strip('`').replace('\n', '')[4:]
  claims_dict = json.loads(clean_json_str)
  print(type(claims_dict))
  print(claims_dict)
  return claims_dict
  #print(text)
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
