# Umu nlp lab 3?

## how to run

for the different parts of the labs run

```
uv run run_*.py
```

for `instruct_llm.py` first run the vllm server

## Prerequisites

- uv
- NVIDIA GPU with CUDA support (for vLLM)

## Lessons learned

- Increase batch_size and learning rate proportionally
- Keep models small
- Do one change at a time
- Always run an experiment while refactoring other code
- tfidf still beat everyone else
- One difficult class (19) is making it hard for models.

## tokenization and preparing data

- Actually study the data first and determine good parameters to make good
  inferences like vocab size

## Modularization of pytorch

- Very nice

## class imbalance

- use label smoothing in cross entropy

## NN regularization techniques

- drop out
- layer normalization

## Running instruct_llm.py

The `instruct_llm.py` script evaluates sentiment classification using different
prompting strategies (zero-shot, one-shot, XML-formatted) with a locally hosted
LLM.

### 1. Start the vLLM server

First, start the vLLM server hosting the Qwen model:

```bash
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name qwen-instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

This will:

- Download the Qwen3-4B-Instruct model from Hugging Face (first time only)
- Start an OpenAI-compatible API server on http://localhost:8000
- Use 90% of available GPU memory
- Set max context length to 4096 tokens

### 2. Run the evaluation script

In a separate terminal, run the sentiment classification evaluation:

```bash
uv run python instruct_llm.py
```

The script will:

- Load training and dev data from `reviews-train.txt` and `reviews-dev.txt`
- Send classification requests to the local vLLM server
- Calculate F1 score, precision, recall, and error rate
