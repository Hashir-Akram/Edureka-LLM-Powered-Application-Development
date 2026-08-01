import requests
import json
import time

# ── CONFIGURATION ─────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"   # Ollama runs here by default
MODEL_NAME = "gemma3:1b"                 # Change to any model you've pulled

# ── HEALTH CHECK ──────────────────────────────────────────────────
def check_ollama_running():
    """Make sure Ollama is running before we try to use it."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_available_models():
    """See what models you have downloaded locally."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


# ── CORE API CALL FUNCTION ────────────────────────────────────────
def ask_ollama(prompt: str, system: str = "", stream: bool = True) -> str:
    """
    Send a prompt to your local Ollama model.

    HOW THIS WORKS:
    Ollama provides an HTTP API on port 11434.
    We send a POST request with our prompt.
    The model processes it ENTIRELY on your CPU/GPU — no internet.
    The response streams back token by token (just like ChatGPT).

    stream=True  → See words appear one by one (more natural)
    stream=False → Wait for full response, then print all at once
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system,
        "stream": stream
    }

    start = time.time()
    full_response = ""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        stream=stream
    )

    if stream:
        print("", end="", flush=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if data.get("done", False):
                    break
        print()  # new line after streaming
    else:
        data = response.json()
        full_response = data.get("response", "")

    elapsed = round(time.time() - start, 2)
    return full_response, elapsed


# ── DEMO SCENARIOS ────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  🏠 DEMO 4 — Local LLM with Ollama")
    print("  100% on your machine. No internet. No API key. Free forever.")
    print("=" * 60)

    # Check if Ollama is running
    if not check_ollama_running():
        print("\n❌ Ollama is not running!")
        print("   Fix: Open a terminal and run: ollama serve")
        print("   Then: ollama pull gemma3:1b")
        print("   Then re-run this file.")
        return

    # Show available models
    models = list_available_models()
    print(f"\n✅ Ollama is running!")
    print(f"   Available models on your machine: {models if models else ['none — run: ollama pull gemma3:1b']}")
    print(f"   Currently using: {MODEL_NAME}")
    print()

    # ── SCENARIO 1: Simple Question ──────────────────────────────
    print("─" * 60)
    print("SCENARIO 1: Simple Question (same as any cloud API)")
    print("─" * 60)
    print("Prompt: 'What is quantization in AI?'")
    print("Answer: ", end="")

    answer, t = ask_ollama(
        prompt="What is quantization in AI? Explain in 2 sentences.",
        stream=True
    )
    print(f"⏱ Time taken: {t}s (running on YOUR CPU/GPU)\n")

    # ── SCENARIO 2: Private Data — The Key Use Case ───────────────
    print("─" * 60)
    print("SCENARIO 2: Private/Sensitive Data (WHERE LOCAL SHINES)")
    print("─" * 60)
    print("Prompt: Analyzing private medical notes...")
    print()

    private_prompt = """
    Patient note (CONFIDENTIAL):
    Patient: John D., 45 years old
    Symptoms: Fatigue, mild chest discomfort, shortness of breath on exertion.
    Current meds: Metformin 500mg, Atorvastatin 20mg
    
    Task: Summarize the key clinical concerns in bullet points.
    """
    print("Summary: ", end="")
    answer, t = ask_ollama(
        prompt=private_prompt,
        system="You are a clinical documentation assistant. Be precise and factual.",
        stream=True
    )
    print(f"⏱ Time: {t}s")
    print("✅ THIS DATA NEVER LEFT YOUR MACHINE. Perfect for healthcare, legal, banking.\n")

    # ── SCENARIO 3: Code Generation ───────────────────────────────
    print("─" * 60)
    print("SCENARIO 3: Code Generation (great for developers)")
    print("─" * 60)
    print("Prompt: Write a Python function...")
    print()

    answer, t = ask_ollama(
        prompt="Write a Python function to validate an email address using regex. Include 3 test cases.",
        system="You are a Python expert. Write clean, commented code.",
        stream=True
    )
    print(f"⏱ Time: {t}s\n")

    # ── INTERACTIVE MODE ──────────────────────────────────────────
    print("=" * 60)
    print("  💬 INTERACTIVE MODE — Talk to your local model!")
    print("  (Type 'quit' to exit)")
    print("=" * 60)
    print()

    system_prompt = "You are a helpful local AI assistant. Be concise and friendly."
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        # Build context from history
        context = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history[-4:]])
        full_prompt = f"{context}\nUSER: {user_input}\nASSISTANT:" if context else user_input

        print("Local AI: ", end="")
        answer, t = ask_ollama(full_prompt, system=system_prompt, stream=True)
        print(f"⏱ {t}s | Running 100% locally\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

    print("\n✅ Demo complete!")
    print("   Everything you just did ran entirely on your computer.")
    print("   No data was sent anywhere. No API costs. No rate limits.")


if __name__ == "__main__":
    main()
