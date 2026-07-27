# Edureka LLM-Powered Application Development

Hands-on demos exploring how to build applications on top of Large Language Models (LLMs), using [Groq](https://console.groq.com/) as an OpenAI-compatible inference provider for fast, open-source model access.

## Contents

| File | Description |
|---|---|
| [demo1_first_api_call.py](demo1_first_api_call.py) | First LLM API call — a minimal chat completion request against LLaMA 3.3 70B. |
| [demo2_compare_models.py](demo2_compare_models.py) | Compares open-source models (LLaMA, GPT-OSS, Qwen) across reasoning, creative writing, code generation, and summarization tasks, benchmarking speed and token usage. |
| [Introduction to LLM.pdf](Introduction%20to%20LLM.pdf) | Course reference slides. |

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

## Running the demos

```bash
python demo1_first_api_call.py
python demo2_compare_models.py
```

## Requirements

- Python 3.9+
- A [Groq](https://console.groq.com/) API key (free tier available)
