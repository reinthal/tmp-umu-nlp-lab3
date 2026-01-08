import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gensim.downloader as api
from torch.nn.functional import cosine_similarity
import pickle
import os

# ----------------------------
# Pooling utilities
# ----------------------------
def mean_pooling_static(token_embeddings: torch.Tensor) -> torch.Tensor:
    return token_embeddings.mean(dim=0)

def minmax_pooling_static(token_embeddings: torch.Tensor) -> torch.Tensor:
    max_pooled = token_embeddings.max(dim=0)[0]
    min_pooled = token_embeddings.min(dim=0)[0]
    return torch.cat([max_pooled, min_pooled], dim=0)

# ----------------------------
# Contextual embeddings
# ----------------------------
def get_sentence_embedding(sentence, model, tokenizer, pooling="average"):
    token_ids = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**token_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        # Convert to float32 to avoid overflow in pooling operations
        hidden_states = hidden_states.float()
        attention_mask = token_ids["attention_mask"].unsqueeze(-1)

        if pooling == "average":
            masked = hidden_states * attention_mask
            embedding = masked.sum(dim=1) / attention_mask.sum(dim=1)

        elif pooling == "minmax":
            hidden_max = hidden_states.masked_fill(attention_mask == 0, float("-inf"))
            max_pooled = hidden_max.max(dim=1)[0]

            hidden_min = hidden_states.masked_fill(attention_mask == 0, float("inf"))
            min_pooled = hidden_min.min(dim=1)[0]

            embedding = torch.cat([max_pooled, min_pooled], dim=1)

        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    # Return: pooled embedding, token_ids, and token-level embeddings for bootstrapping
    token_embeddings = hidden_states.squeeze(0)  # (seq_len, hidden_dim)
    attention_mask_1d = token_ids["attention_mask"].squeeze(0)  # (seq_len,)
    # Filter out padding tokens
    token_embeddings_valid = token_embeddings[attention_mask_1d == 1]

    return embedding.squeeze(0), token_ids, token_embeddings_valid

# ----------------------------
# Static GloVe embeddings
# ----------------------------
def get_glove_embedding(token_ids, tokenizer, glove_model):
    tokens = []
    embeddings = []

    input_ids = token_ids["input_ids"][0]
    attention_mask = token_ids["attention_mask"][0]

    for i in tqdm(range(input_ids.shape[0]), desc="GloVe tokens", leave=False):
        if attention_mask[i] == 0:
            break

        token_text = tokenizer.decode([input_ids[i].item()]).strip().lower()
        tokens.append(token_text)

        if token_text in glove_model:
            embeddings.append(glove_model[token_text])
        else:
            embeddings.append(np.zeros(glove_model.vector_size))

    return torch.tensor(np.array(embeddings), dtype=torch.float32), tokens

# ----------------------------
# Bootstrap similarity
# ----------------------------
def bootstrap_similarity(emb1: torch.Tensor, emb2: torch.Tensor, n_boot=1000, pooling="average"):
    """
    Compute 95% CI for cosine similarity between two embeddings via bootstrapping.
    Works for both 1D and 2D (token-level) embeddings.
    """
    sims = []

    # If embeddings are 1D, replicate to 2D
    if emb1.dim() == 1:
        emb1_2d = emb1.unsqueeze(0)
    else:
        emb1_2d = emb1
    if emb2.dim() == 1:
        emb2_2d = emb2.unsqueeze(0)
    else:
        emb2_2d = emb2

    n_tokens = min(emb1_2d.shape[0], emb2_2d.shape[0])
    if n_tokens == 0:
        raise ValueError("No tokens available for bootstrapping.")

    for _ in tqdm(range(n_boot), desc=f"Bootstrapping {pooling}"):
        idx = np.random.choice(n_tokens, n_tokens, replace=True)
        sample1 = emb1_2d[idx]
        sample2 = emb2_2d[idx]

        if pooling == "average":
            pooled1 = sample1.mean(dim=0)
            pooled2 = sample2.mean(dim=0)
        elif pooling == "minmax":
            pooled1 = torch.cat([sample1.max(dim=0)[0], sample1.min(dim=0)[0]], dim=0)
            pooled2 = torch.cat([sample2.max(dim=0)[0], sample2.min(dim=0)[0]], dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Ensure 2D for cosine_similarity
        pooled1 = pooled1.view(1, -1)
        pooled2 = pooled2.view(1, -1)

        sim = cosine_similarity(pooled1, pooled2).item()
        sims.append(sim)

    return np.percentile(sims, 2.5), np.percentile(sims, 97.5), np.mean(sims)

# ----------------------------
# Main
# ----------------------------
def main():
    # ---- Load data ----
    with open("data/dataset.0", "r") as f:
        dataset_0 = f.read()
    with open("data/dataset.1", "r") as f:
        dataset_1 = f.read()

    # ---- Check for cached embeddings ----
    cache_file = "pooling2_cache.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
            emb0_avg = cache['emb0_avg']
            emb1_avg = cache['emb1_avg']
            emb0_minmax = cache['emb0_minmax']
            emb1_minmax = cache['emb1_minmax']
            emb0_tokens = cache['emb0_tokens']
            emb1_tokens = cache['emb1_tokens']
            glv0_emb = cache['glv0_emb']
            glv1_emb = cache['glv1_emb']
            glv0_avg = cache['glv0_avg']
            glv1_avg = cache['glv1_avg']
            glv0_minmax = cache['glv0_minmax']
            glv1_minmax = cache['glv1_minmax']
        print("Loaded cached embeddings successfully!")
    else:
        # ---- Load models ----
        print("Loading GloVe...")
        glove_300 = api.load("glove-wiki-gigaword-300")

        print("Loading Qwen model...")
        model_name = "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
        )
        model.eval()

        # ---- Contextual embeddings ----
        print("Computing contextual embeddings...")
        emb0_avg, token_ids_0, emb0_tokens = get_sentence_embedding(dataset_0, model, tokenizer, pooling="average")
        emb1_avg, token_ids_1, emb1_tokens = get_sentence_embedding(dataset_1, model, tokenizer, pooling="average")

        emb0_minmax, _, _ = get_sentence_embedding(dataset_0, model, tokenizer, pooling="minmax")
        emb1_minmax, _, _ = get_sentence_embedding(dataset_1, model, tokenizer, pooling="minmax")

        # ---- Static GloVe embeddings ----
        print("Computing static GloVe embeddings...")
        glv0_emb, _ = get_glove_embedding(token_ids_0, tokenizer, glove_300)
        glv1_emb, _ = get_glove_embedding(token_ids_1, tokenizer, glove_300)

        # Apply pooling
        glv0_avg = mean_pooling_static(glv0_emb)
        glv1_avg = mean_pooling_static(glv1_emb)

        glv0_minmax = minmax_pooling_static(glv0_emb)
        glv1_minmax = minmax_pooling_static(glv1_emb)

        # ---- Cache embeddings ----
        print(f"Saving embeddings to {cache_file}...")
        cache = {
            'emb0_avg': emb0_avg,
            'emb1_avg': emb1_avg,
            'emb0_minmax': emb0_minmax,
            'emb1_minmax': emb1_minmax,
            'emb0_tokens': emb0_tokens,
            'emb1_tokens': emb1_tokens,
            'glv0_emb': glv0_emb,
            'glv1_emb': glv1_emb,
            'glv0_avg': glv0_avg,
            'glv1_avg': glv1_avg,
            'glv0_minmax': glv0_minmax,
            'glv1_minmax': glv1_minmax,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        print("Embeddings cached successfully!")

    # ---- Similarity scores ----
    score1 = cosine_similarity(emb0_avg.unsqueeze(0), emb1_avg.unsqueeze(0)).item()
    score2 = cosine_similarity(emb0_minmax.unsqueeze(0), emb1_minmax.unsqueeze(0)).item()
    score3 = cosine_similarity(glv0_avg.unsqueeze(0), glv1_avg.unsqueeze(0)).item()
    score4 = cosine_similarity(glv0_minmax.unsqueeze(0), glv1_minmax.unsqueeze(0)).item()

    print("\n===== Cosine Similarity Scores =====")
    print(f"Score1 (contextual + average pooling): {score1:.4f}")
    print(f"Score2 (contextual + min-max pooling): {score2:.4f}")
    print(f"Score3 (static + average pooling):     {score3:.4f}")
    print(f"Score4 (static + min-max pooling):     {score4:.4f}")

    # ---- Bootstrap 95% confidence intervals ----
    # Use token-level embeddings for bootstrapping, NOT pooled embeddings
    print("\n===== 95% Confidence Intervals via Bootstrapping =====")
    ci1 = bootstrap_similarity(emb0_tokens, emb1_tokens, pooling="average")
    ci2 = bootstrap_similarity(emb0_tokens, emb1_tokens, pooling="minmax")
    ci3 = bootstrap_similarity(glv0_emb, glv1_emb, pooling="average")
    ci4 = bootstrap_similarity(glv0_emb, glv1_emb, pooling="minmax")

    print(f"Score1 CI:  {ci1[0]:.4f} - {ci1[1]:.4f} (mean={ci1[2]:.4f})")
    print(f"Score2 CI:  {ci2[0]:.4f} - {ci2[1]:.4f} (mean={ci2[2]:.4f})")
    print(f"Score3 CI:  {ci3[0]:.4f} - {ci3[1]:.4f} (mean={ci3[2]:.4f})")
    print(f"Score4 CI:  {ci4[0]:.4f} - {ci4[1]:.4f} (mean={ci4[2]:.4f})")

if __name__ == "__main__":
    main()
