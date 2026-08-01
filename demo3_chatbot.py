import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── CHOOSE YOUR CHATBOT PERSONA ───────────────────────────────────
# Change this system prompt to completely change the bot's personality
SYSTEM_PROMPT = """You are ARIA — a friendly, professional AI assistant 
for TechStart company. You help employees with HR questions, IT support, 
and general company information.

Rules:
- Keep responses under 3 sentences unless asked for more
- If you don't know something specific, say so honestly
- Always be warm and professional
- Never make up company policies — say "I'll check on that for you"

Company info you know:
- Office hours: 9am to 6pm IST, Monday to Friday
- IT helpdesk: ext. 1234 or it-help@techstart.com
- HR team: ext. 5678 or hr@techstart.com
"""

def chat(conversation_history: list, user_message: str) -> str:
    """
    The core chat function. Takes the FULL history + new message.
    Returns the assistant's reply AND adds everything to history.

    WHY WE PASS THE FULL HISTORY:
    LLMs have no memory between calls. They are stateless.
    Every call is fresh. To simulate memory, we manually include
    the entire conversation in every API request.
    This is how EVERY chatbot works — ChatGPT, Claude, all of them.
    """
    # Add the new user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Send the FULL history to the model every single time
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + conversation_history,   # ← entire conversation every time
        temperature=0.5,
        max_tokens=300
    )

    assistant_reply = response.choices[0].message.content

    # Add the assistant's reply to history too
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return assistant_reply


def run_chatbot():
    """Run the interactive chatbot in the terminal."""
    conversation_history = []   # starts empty — no memory yet

    print("\n" + "=" * 60)
    print("  🤖 ARIA — TechStart AI Assistant")
    print("  Powered by LLaMA 3.3 70B via Groq")
    print("  Type 'quit' to exit | Type 'history' to see memory")
    print("=" * 60)
    print()
    print("ARIA: Hello! I'm ARIA, TechStart's AI assistant. How can I help you today?")
    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n\nARIA: Goodbye! Have a great day! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("\nARIA: Thank you for chatting! Have a wonderful day! 👋")
            break

        # Special command: show the conversation history
        if user_input.lower() == 'history':
            print("\n📜 CONVERSATION MEMORY (what the model sees every time):")
            print(f"   Messages in memory: {len(conversation_history)}")
            for i, msg in enumerate(conversation_history, 1):
                role = "You" if msg["role"] == "user" else "ARIA"
                print(f"   [{i}] {role}: {msg['content'][:60]}...")
            print()
            continue

        # Get response from the model
        print("\nARIA: ", end="", flush=True)
        response = chat(conversation_history, user_input)
        print(response)
        print(f"      [Memory: {len(conversation_history)} messages stored]\n")


# ── AUTOMATED DEMO (for classroom / quick testing) ────────────────
def run_automated_demo():
    """
    Run a scripted conversation to show chatbot memory working.
    Perfect for live classroom demo without typing.
    """
    print("\n" + "=" * 60)
    print("  🎭 AUTOMATED DEMO — Showing how chatbot memory works")
    print("=" * 60)

    conversation_history = []

    # Scripted conversation that tests memory
    demo_exchanges = [
        "Hi! My name is Priya and I'm a new employee.",
        "What are the office hours?",
        "Can you remind me what my name is?",      # Tests memory
        "Who should I contact for IT help?",
        "Actually, can you summarize everything useful you've told me so far?"  # Tests full memory
    ]

    print("\nARIA: Hello! I'm ARIA. How can I help you today?")
    print()

    for user_msg in demo_exchanges:
        print(f"You:  {user_msg}")
        response = chat(conversation_history, user_msg)
        print(f"ARIA: {response}")
        print(f"      [📧 Memory size: {len(conversation_history)} messages]\n")

    print("=" * 60)
    print("KEY LEARNING:")
    print("  When asked 'What is my name?' — ARIA remembered 'Priya'")
    print("  because the FULL conversation was sent with every API call.")
    print("  Without this, EVERY question would feel like starting fresh.")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_automated_demo()
    else:
        run_chatbot()
