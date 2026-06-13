import os
from dotenv import load_dotenv
import requests
from groq import Groq

load_dotenv()


OLLAMA_URL = os.environ.get('OLLAMA_URL')
MODEL = os.environ.get('MODEL') 
ENABLE_GROQ = os.environ.get('ENABLE_GROQ', 'false').lower() == 'true'


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "stop": ["\nObservation:", "Observation:"]
    }
    if (ENABLE_GROQ) :
        print("groq is running")
        client = Groq()
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content
    else:
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "stop": ["\nObservation:", "Observation:"] 
            }
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]