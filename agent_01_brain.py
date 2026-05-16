import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()


OLLAMA_URL = os.environ.get('OLLAMA_URL')
MODEL = os.environ.get('MODEL') 


print()

def chat(messages: list[dict]) -> dict:
    """Send messages to Ollama and get a response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False  # we want the full response at once
    }

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    elapsed = time.time() - start

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

    data = response.json()

    return {
        "content": data["message"]["content"],
        "model": data["model"],
        "total_duration_sec": round(elapsed, 2),
        "prompt_tokens": data.get("prompt_eval_count", "?"),
        "response_tokens": data.get("eval_count", "?"),
    }

def main():
    print(f"🧠 Talking to {MODEL} via Ollama")
    print("Type 'quit' to exit, 'reset' to start fresh\n")

    # conversation history — this is how the LLM remembers context
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "When answering, first briefly state your reasoning, "
                "then give your final answer clearly."
            )
        }
    ]

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            messages = [messages[0]]  # keep system prompt, clear history
            print("🔄 Conversation reset.\n")
            continue
        
        # add user message to history
        messages.append({"role": "user", "content": user_input})

        print("🤔 Thinking...", end="\r")

        try:
            result = chat(messages)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        assistant_reply = result["content"]

        # add assistant reply to history (this is how multi-turn works)
        messages.append({"role": "assistant", "content": assistant_reply})


        print(f"\n🤖 Assistant:\n{assistant_reply}")
        print(f"\n📊 [{result['total_duration_sec']}s | "
              f"prompt={result['prompt_tokens']} tokens | "
              f"response={result['response_tokens']} tokens | "
              f"history={len(messages)} messages]\n")



if __name__ == "__main__":
    main()