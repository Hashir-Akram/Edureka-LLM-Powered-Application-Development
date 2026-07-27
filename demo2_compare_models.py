import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── MODELS TO COMPARE ────────────────────────────────────────────
MODELS = {
    "Llama 3.1 8B (Meta)":     "llama-3.1-8b-instant",
    "GPT-OSS 20B (OpenAI)":    "openai/gpt-oss-20b",
    "Qwen 3.6 27B (Alibaba)":  "qwen/qwen3.6-27b",
}

# ── TEST QUESTIONS (one per category) ────────────────────────────
TEST_QUESTIONS = [
    {
        "category": "Reasoning & Logic",
        "question": "A train leaves Delhi at 8am going 60km/h. Another leaves Mumbai at 9am going 90km/h toward Delhi. The cities are 1400km apart. When do they meet? Show your working briefly."
    },
    {
        "category": "Creative Writing",
        "question": "Write a 3-line product tagline for an AI-powered doctor appointment app targeting patients in rural India."
    },
    {
        "category": "Code Generation",
        "question": "Write a Python function that takes a list of numbers and returns the top 3 unique values. Include one example."
    },
    {
        "category": "Summarization",
        "question": "In 2 sentences, explain why quantization makes AI models faster and cheaper to run."
    }
]

def call_model(model_id, question, system_prompt="Be clear and helpful."):
    """Call a model and measure time + tokens."""
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=300
    )
    elapsed = round(time.time() - start, 2)
    return {
        "answer": response.choices[0].message.content,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
        "time_s": elapsed,
        "speed": round(response.usage.completion_tokens / elapsed, 1)  # tokens/sec
    }

print("=" * 70)
print("  DEMO 2 — Open-Source Model Comparison")
print("  Testing LLaMA vs GPT-OSS vs Qwen on real tasks")
print("=" * 70)

# ── RUN EACH TEST ────────────────────────────────────────────────
results_summary = []

for test in TEST_QUESTIONS:
    print(f"\n{'─'*70}")
    print(f"📋 CATEGORY: {test['category']}")
    print(f"❓ QUESTION: {test['question'][:80]}...")
    print(f"{'─'*70}")

    question_results = {"category": test["category"], "models": {}}

    for model_name, model_id in MODELS.items():
        try:
            result = call_model(model_id, test["question"])
            question_results["models"][model_name] = result

            print(f"\n🤖 {model_name}")
            print(f"   ⏱  Time: {result['time_s']}s  |  Speed: {result['speed']} tok/s  |  Output tokens: {result['tokens_out']}")
            print(f"   Answer: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        except Exception as e:
            print(f"\n🤖 {model_name}")
            print(f"   ⚠️ Error: {e}")

    results_summary.append(question_results)

# ── FINAL COMPARISON TABLE ────────────────────────────────────────
print(f"\n{'='*70}")
print("  📊 SPEED COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Category':<25} {'LLaMA Speed':>15} {'GPT-OSS Speed':>15} {'Qwen Speed':>14}")
print(f"{'─'*70}")

for result in results_summary:
    cat = result["category"][:24]
    speeds = []
    for model_name in MODELS.keys():
        if model_name in result["models"]:
            speeds.append(f"{result['models'][model_name]['speed']:>13.1f} t/s")
        else:
            speeds.append(f"{'N/A':>13}")
    print(f"{cat:<25} {speeds[0]:>15} {speeds[1]:>15} {speeds[2]:>14}")

print(f"\n{'='*70}")
print("  🎓 WHAT TO PICK AND WHEN:")
print(f"{'='*70}")
print("  LLaMA 3.3 8B  → Best for complex reasoning, coding, long documents")
print("  GPT-OSS 20B  → Best for speed-sensitive apps, chat, fast responses")
print("  Qwen 3.6 27B     → Best for safe outputs, education, content for minors")
print()
print("  💡 TIP: Run this again with a question from YOUR industry")
print("     and see which model gives the best answer for YOUR use case.")
print("     Benchmarks on slides are fine — your own test is BETTER.")
