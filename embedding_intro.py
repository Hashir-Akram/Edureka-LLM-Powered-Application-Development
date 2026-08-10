"""
  pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ── LOAD THE EMBEDDING MODEL ──────────────────────────────────────────────────
# This downloads the model once (~90MB) and caches it locally.
# Every sentence you pass in → comes back as a list of 384 numbers.
print("\n⏳ Loading embedding model (downloads once, ~90MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded!\n")


def get_embedding(text: str) -> np.ndarray:
    """
    Convert any text into a vector (list of numbers).

    This is the core operation that makes semantic search possible.
    The model was trained on billions of sentences so it 'knows'
    which words/sentences have similar meanings.

    Returns: numpy array of shape (384,) — 384 float numbers.
    """
    return model.encode(text, normalize_embeddings=True)
    # normalize_embeddings=True → scales each vector to length 1
    # This makes cosine similarity = dot product (faster math)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Measure how similar two vectors are.

    Score of 1.0 = identical meaning
    Score of 0.5 = somewhat related
    Score of 0.0 = completely unrelated
    Score < 0    = opposite meaning (rare)

    Math: cosine of the angle between two vectors.
    Since we normalized, this is just the dot product.
    """
    return float(np.dot(vec1, vec2))


def similarity_label(score: float) -> str:
    """Human-readable label for a similarity score."""
    if score > 0.85: return "🟢 VERY SIMILAR"
    if score > 0.65: return "🟡 SOMEWHAT SIMILAR"
    if score > 0.40: return "🟠 LOOSELY RELATED"
    return "🔴 UNRELATED"


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — SEE WHAT AN EMBEDDING LOOKS LIKE
# ══════════════════════════════════════════════════════════════════════════════

def demo_what_is_an_embedding():
    print("=" * 68)
    print("  PART 1 — What Does an Embedding Actually Look Like?")
    print("=" * 68)

    text = "The dog ran fast across the park"
    vector = get_embedding(text)

    print(f"\n  Input text:   \"{text}\"")
    print(f"\n  Output vector (first 10 of 384 numbers):")
    print(f"  {vector[:10].round(4).tolist()}")
    print(f"\n  Full vector shape: {vector.shape}  ← 384 numbers in total")
    print(f"  Vector magnitude:  {np.linalg.norm(vector):.4f}  ← always 1.0 when normalized")

    print(f"""
  WHAT THESE NUMBERS MEAN:
  ─────────────────────────────────────────────────────────────
  Each of these 384 numbers represents a learned 'feature'
  of the text. Position 0 might loosely encode 'animal-ness',
  position 1 might encode 'motion', and so on.

  The model learned these features from billions of sentences.
  You don't need to understand each individual number —
  what matters is that sentences with SIMILAR MEANINGS will
  produce vectors with SIMILAR NUMBERS.
  ─────────────────────────────────────────────────────────────
""")
    input("  Press ENTER for Part 2...\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — THE KEY INSIGHT: SIMILAR MEANING = SIMILAR NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

def demo_similarity_insight():
    print("=" * 68)
    print("  PART 2 — The Key Insight: Similar Meaning = Similar Numbers")
    print("=" * 68)
    print("""
  We're going to take ONE sentence and compare it to many others.
  Watch how the similarity score changes based on MEANING — not words.

  Reference sentence: "The dog ran fast"
""")

    reference = "The dog ran fast"
    ref_vector = get_embedding(reference)

    comparisons = [
        # (sentence, reason it's being tested)
        ("The puppy sprinted quickly",         "Different words, same meaning"),
        ("A fast-running canine",              "Very different words, same concept"),
        ("The cat moved swiftly",              "Different animal, same action"),
        ("The dog walked slowly",              "Same subject, opposite action"),
        ("I love eating pizza on weekends",    "Completely different topic"),
        ("Stock markets fell sharply today",   "Totally unrelated domain"),
        ("Quantum entanglement is fascinating","Completely different field"),
    ]

    print(f"  {'Comparison Sentence':<42} {'Score':>6}  {'Label'}")
    print(f"  {'─'*41} {'─'*6}  {'─'*20}")

    for sentence, reason in comparisons:
        vec    = get_embedding(sentence)
        score  = cosine_similarity(ref_vector, vec)
        label  = similarity_label(score)
        print(f"  \"{sentence[:40]}\"")
        print(f"  {'':>2} Why: {reason}")
        print(f"  {'':>2} Score: {score:.4f}  {label}\n")

    print("  KEY OBSERVATION:")
    print("  'The puppy sprinted quickly' scores HIGH even though")
    print("  it shares ZERO words with 'The dog ran fast'.")
    print("  This is semantic similarity — and it's impossible with keyword search.")
    input("\n  Press ENTER for Part 3...\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — THE BUSINESS SCENARIO: WHY THIS MATTERS
# ══════════════════════════════════════════════════════════════════════════════

def demo_business_scenario():
    print("=" * 68)
    print("  PART 3 — Real Business Scenario: Customer Support Search")
    print("=" * 68)
    print("""
  SCENARIO: You work at an e-commerce company.
  You have a help center with 10 articles.
  A customer types a question — can you find the right article?

  CHALLENGE: The customer's words will NOT match the article titles.
  This is the exact problem that beats keyword search every time.
""")

    # The knowledge base (article titles)
    articles = [
        "How to cancel a duplicate order",
        "Returning a damaged or defective product",
        "Tracking your shipment in real time",
        "Updating your payment method",
        "How to apply a promo code or discount",
        "Changing your delivery address after ordering",
        "What to do if your package is lost",
        "Contacting customer support by phone or chat",
        "How to leave a product review",
        "Account security and password reset",
    ]

    # Customer queries — notice the words DON'T match article titles
    customer_queries = [
        "I accidentally bought two of the same thing",       # → should find: cancel duplicate
        "My item arrived broken",                            # → should find: damaged product
        "Where is my order right now?",                      # → should find: tracking shipment
        "I put the wrong address when I checked out",        # → should find: change delivery address
    ]

    # Embed all articles once
    print("  Embedding knowledge base articles...")
    article_vectors = [get_embedding(a) for a in articles]
    print("  ✅ Done.\n")

    for query in customer_queries:
        query_vector = get_embedding(query)

        # Find most similar article
        scores = [cosine_similarity(query_vector, av) for av in article_vectors]
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        print(f"  Customer query:   \"{query}\"")
        print(f"  Best match:       \"{articles[best_idx]}\"")
        print(f"  Similarity score: {best_score:.4f}  {similarity_label(best_score)}")
        print()

    print("  NOTICE: Not a single query used the exact words from the article title.")
    print("  But the embedding model found the right article every time.")
    print()
    print("  This is the foundation of RAG — and now you understand why it works.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧠 DEMO 4A — Embeddings: Turning Words Into Numbers")
    print("   Model: all-MiniLM-L6-v2 (HuggingFace, free, local)")
    print("   No API key needed. Everything runs on your machine.\n")

    demo_what_is_an_embedding()
    demo_similarity_insight()
    demo_business_scenario()

    print("\n" + "=" * 68)
    print("  ✅ DEMO 4A COMPLETE")
    print()
    print("  What you learned:")
    print("    → An embedding is a list of ~384 numbers representing meaning")
    print("    → Cosine similarity measures how close two meanings are (0–1)")
    print("    → Similar meanings → similar numbers (even with different words)")
    print("    → This is the foundation of semantic search and RAG")
    print()
    print("  NEXT: Run demo4b_vector_search_faiss.py to build a real search engine")
    print("=" * 68 + "\n")