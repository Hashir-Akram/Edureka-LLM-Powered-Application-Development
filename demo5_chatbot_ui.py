import streamlit as st
import requests
import json
import time

# ── APP CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

OLLAMA_URL = "http://localhost:11434"

# ── HELPER FUNCTIONS ──────────────────────────────────────────────
def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

def ollama_chat(messages: list, model: str, system: str) -> str:
    """
    Send the FULL conversation history to Ollama and get a response.

    FIX EXPLAINED:
    Previously the caller passed messages[:-1] — excluding the user's
    latest question. The model never saw what was asked, so it responded
    with a clarifying question instead of an answer.

    Now we always pass the complete messages list here. The last item
    in the list IS the user's current question — and the model needs it.
    """
    prompt_parts = []
    for msg in messages:
        if msg["role"] == "user":
            prompt_parts.append(f"USER: {msg['content']}")
        else:
            prompt_parts.append(f"ASSISTANT: {msg['content']}")
    prompt_parts.append("ASSISTANT:")

    payload = {
        "model": model,
        "system": system,
        "prompt": "\n".join(prompt_parts),
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        return response.json().get("response", "Error: No response from model")
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ── SIDEBAR CONFIGURATION ─────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    available_models = get_available_models()
    if available_models:
        selected_model = st.selectbox("🤖 Select Model", available_models)
        st.success(f"✅ Ollama running")
    else:
        selected_model = "llama3.2"
        st.error("❌ Ollama not running")
        st.code("# Start with:\nollama serve\nollama pull llama3.2")

    st.divider()
    st.subheader("🎭 Choose Chatbot Persona")
    persona = st.selectbox("Persona", [
        "General Assistant",
        "Python Coding Expert",
        "Medical Assistant (Safe)",
        "HR Helpdesk",
        "Creative Writer",
        "Explain Like I'm 10"
    ])

    system_prompts = {
        "General Assistant": "You are a helpful, friendly AI assistant. Be concise and clear.",
        "Python Coding Expert": "You are a Python expert. Always provide working code with comments. Mention edge cases.",
        "Medical Assistant (Safe)": "You are a medical information assistant. Provide factual health information but always advise consulting a doctor for personal medical decisions.",
        "HR Helpdesk": "You are an HR assistant. Help with workplace queries professionally. For sensitive issues, recommend speaking with an HR manager.",
        "Creative Writer": "You are a creative writing assistant. Be imaginative, use vivid descriptions, and help craft engaging stories.",
        "Explain Like I'm 10": "Explain everything as if talking to a 10-year-old child. Use simple words, fun analogies, and real-world examples."
    }
    system_prompt = system_prompts[persona]

    st.divider()
    st.caption("💡 This chatbot runs 100% on your local machine.")
    st.caption("No internet needed after model download.")
    st.caption("Your data never leaves your computer.")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if "messages" in st.session_state:
        st.metric("Messages in memory", len(st.session_state.messages))

# ── MAIN CHAT INTERFACE ───────────────────────────────────────────
st.title("🤖 Local AI Chatbot")
st.caption(f"Running {selected_model} via Ollama | 100% Local | No API Key")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(f"Hello! I'm your local AI assistant running as a **{persona}**. How can I help you today?")

# ── HANDLE USER INPUT ─────────────────────────────────────────────
if prompt := st.chat_input("Type your message here..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Add to history FIRST — so it's included in the API call
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response — pass the FULL messages list (including the question just added)
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 {selected_model} is thinking..."):
            start_time = time.time()
            response = ollama_chat(
                st.session_state.messages,      # ✅ FIXED: full list, not [:-1]
                selected_model,
                system_prompt
            )
            elapsed = round(time.time() - start_time, 1)

        st.write(response)
        st.caption(f"⏱ {elapsed}s | Model: {selected_model} | Running locally")

    # Add assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# ── QUICK DEMO PROMPTS ────────────────────────────────────────────
st.divider()
st.subheader("💡 Try These Quick Demos")

demo_prompts = {
    "🔬 Science": "Explain how vaccines work in simple terms",
    "💻 Code": "Write a Python function to count word frequency in a text",
    "📊 Business": "What are the top 3 ways AI can reduce customer service costs?",
    "🎨 Creative": "Write a short poem about artificial intelligence and humanity",
    "🧮 Math": "If I invest ₹10,000 monthly at 12% annual return, how much will I have in 10 years?"
}

cols = st.columns(len(demo_prompts))
for col, (emoji_label, prompt_text) in zip(cols, demo_prompts.items()):
    with col:
        if st.button(emoji_label, use_container_width=True):
            # Add user prompt to history first
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            with st.spinner("Thinking..."):
                response = ollama_chat(
                    st.session_state.messages,  # ✅ FIXED: full list, not [:-1]
                    selected_model,
                    system_prompt
                )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()