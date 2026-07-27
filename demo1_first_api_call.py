import os
from openai import OpenAI
from dotenv import load_dotenv  # Load environment variables from .env file

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")  # Get the API key from environment variables


if not api_key:
    print("❌ No API key found!")
    print("   Create a .env file with: GROQ_API_KEY=your_key_here")
    print("   Get a free key at: https://console.groq.com/keys")
    exit(1)


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


print("=" * 60)
print("  DEMO 1 — Your First LLM API Call")
print("  Model: LLaMA 3.3 70B (Meta's open-source model, hosted on Groq)")
print("=" * 60)
print()


response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # Meta's LLaMA 3.3 — open source!
    messages=[
        {
            "role": "system",           # System message = instructions to the AI
            "content": "You are a friendly AI assistant. Keep answers short and clear."
        },
        {
            "role": "user",             # User message = the actual question
            "content": "Explain what a Large Language Model is in 2 simple sentences."
        }
    ],
    temperature=0.7,    # 0 = very consistent, 1 = more creative
    max_tokens=200      # Maximum words in the response
)

print("🧠 AI Response:")
print(response.choices[0].message.content)