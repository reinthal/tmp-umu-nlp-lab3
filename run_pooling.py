import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gensim.downloader as api
from torch.nn.functional import cosine_similarity
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_similarity_distribution(lower_bound, upper_bound, mean, sims, title):
    """
    Plot histogram of similarity scores with mean and 95% CI overlay.
    
    Parameters:
    -----------
    lower_bound : float
        2.5th percentile (lower bound of 95% CI)
    upper_bound : float
        97.5th percentile (upper bound of 95% CI)
    mean : float
        Mean of similarity scores
    sims : array-like
        Array of all similarity scores
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(sims, bins=30, kde=True, color='skyblue', 
                 edgecolor='black', alpha=0.7, ax=ax)
    
    # Plot mean line
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean:.4f}')
    
    # Plot CI bounds
    ax.axvline(lower_bound, color='orange', linestyle=':', linewidth=2,
               label=f'CI Lower: {lower_bound:.4f}')
    ax.axvline(upper_bound, color='orange', linestyle=':', linewidth=2,
               label=f'CI Upper: {upper_bound:.4f}')
    
    # Shade CI region
    ax.axvspan(lower_bound, upper_bound, alpha=0.2, color='orange',
               label='95% CI')
    
    # Labels and formatting
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


# Example usage for 4 subplots:
def plot_all_similarity_distributions(results_dict):
    """
    Create a 2x2 grid of similarity distributions.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like 'contextual_avg', 'contextual_minmax', etc.
        Each value should be a tuple of (lower_bound, upper_bound, mean, sims)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    titles = [
        'Contextual (Average Pooling)',
        'Contextual (Min-Max Pooling)',
        'Static GloVe (Average Pooling)',
        'Static GloVe (Min-Max Pooling)'
    ]
    
    for idx, (key, title) in enumerate(zip(results_dict.keys(), titles)):
        lower, upper, mean, sims = results_dict[key]
        ax = axes[idx]
        
        # Plot histogram
        sns.histplot(sims, bins=30, kde=True, color='skyblue',
                     edgecolor='black', alpha=0.7, ax=ax)
        
        # Plot mean and CI
        ax.axvline(mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean:.4f}')
        ax.axvline(lower, color='orange', linestyle=':', linewidth=2,
                   label=f'CI: [{lower:.4f}, {upper:.4f}]')
        ax.axvline(upper, color='orange', linestyle=':', linewidth=2)
        ax.axvspan(lower, upper, alpha=0.2, color='orange')
        
        # Formatting
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_overlaid_similarity_distributions(results_dict):
    """
    Create a 1x2 grid comparing contextual vs static embeddings.
    Left: Average Pooling comparison
    Right: Min-Max Pooling comparison
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys: 'contextual_avg', 'contextual_minmax', 
                              'static_avg', 'static_minmax'
        Each value should be a tuple of (lower_bound, upper_bound, mean, sims)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Average Pooling
    ax = axes[0]
    
    # Contextual average
    lower_c, upper_c, mean_c, sims_c = results_dict['contextual_avg']
    sns.histplot(sims_c, bins=30, kde=True, color='steelblue',
                 edgecolor='navy', alpha=0.5, ax=ax, label='Contextual')
    ax.axvline(mean_c, color='darkblue', linestyle='--', linewidth=2,
               label=f'Contextual Mean: {mean_c:.4f}')
    ax.axvspan(lower_c, upper_c, alpha=0.2, color='blue',
               label=f'Contextual 95% CI: [{lower_c:.4f}, {upper_c:.4f}]')
    
    # Static average
    lower_s, upper_s, mean_s, sims_s = results_dict['static_avg']
    sns.histplot(sims_s, bins=30, kde=True, color='coral',
                 edgecolor='darkred', alpha=0.5, ax=ax, label='Static (GloVe)')
    ax.axvline(mean_s, color='darkred', linestyle='--', linewidth=2,
               label=f'Static Mean: {mean_s:.4f}')
    ax.axvspan(lower_s, upper_s, alpha=0.2, color='red',
               label=f'Static 95% CI: [{lower_s:.4f}, {upper_s:.4f}]')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Average Pooling: Contextual vs Static Embeddings', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Min-Max Pooling
    ax = axes[1]
    
    # Contextual minmax
    lower_c, upper_c, mean_c, sims_c = results_dict['contextual_minmax']
    sns.histplot(sims_c, bins=30, kde=True, color='steelblue',
                 edgecolor='navy', alpha=0.5, ax=ax, label='Contextual')
    ax.axvline(mean_c, color='darkblue', linestyle='--', linewidth=2,
               label=f'Contextual Mean: {mean_c:.4f}')
    ax.axvspan(lower_c, upper_c, alpha=0.2, color='blue',
               label=f'Contextual 95% CI: [{lower_c:.4f}, {upper_c:.4f}]')
    
    # Static minmax
    lower_s, upper_s, mean_s, sims_s = results_dict['static_minmax']
    sns.histplot(sims_s, bins=30, kde=True, color='coral',
                 edgecolor='darkred', alpha=0.5, ax=ax, label='Static (GloVe)')
    ax.axvline(mean_s, color='darkred', linestyle='--', linewidth=2,
               label=f'Static Mean: {mean_s:.4f}')
    ax.axvspan(lower_s, upper_s, alpha=0.2, color='red',
               label=f'Static 95% CI: [{lower_s:.4f}, {upper_s:.4f}]')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Min-Max Pooling: Contextual vs Static Embeddings', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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

    return (np.percentile(sims, 2.5), np.percentile(sims, 97.5), np.mean(sims)), sims

# ----------------------------
# Main
# ----------------------------
def main():
    # ---- Load datbootstrap_similaritya ----
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
    print(f"Qwen avg:  (contextual + average pooling): {score1:.4f}")
    print(f"Qwen min/max: (contextual + min-max pooling): {score2:.4f}")
    print(f"GloVe avg:  (static + average pooling):     {score3:.4f}")
    print(f"GloVe min/max: (static + min-max pooling):     {score4:.4f}")

    # ---- Bootstrap 95% confidence intervals ----
    # Use token-level embeddings for bootstrapping, NOT pooled embeddings
    print("\n===== 95% Confidence Intervals via Bootstrapping =====")
    ci_qwen_avg, samples_qwen_avg = bootstrap_similarity(emb0_tokens, emb1_tokens, pooling="average")
    ci_qwen_minmax, samples_qwen_minmax = bootstrap_similarity(emb0_tokens, emb1_tokens, pooling="minmax")
    ci_glove_avg, samples_glove_avg = bootstrap_similarity(glv0_emb, glv1_emb, pooling="average")
    ci_glove_minmax, samples_glove_minmax = bootstrap_similarity(glv0_emb, glv1_emb, pooling="minmax")
    # Multi-plot
    print("\n===== Plotting results =====")
    results = {
        'contextual_avg': (*ci_qwen_avg, samples_qwen_avg),
        'contextual_minmax': (*ci_qwen_minmax, samples_qwen_minmax),
        'static_avg': (*ci_glove_avg, samples_glove_avg),
        'static_minmax': (*ci_glove_minmax, samples_glove_minmax),
    }

    fig = plot_overlaid_similarity_distributions(results)
    plt.savefig('similarity_comparison_2plots.png', dpi=300, bbox_inches='tight')
    print(f"Context (Qwen) avg CI:  {ci_qwen_avg[0]:.4f} - {ci_qwen_avg[1]:.4f} (mean={ci_qwen_avg[2]:.4f})")
    print(f"Context (Qwen) avg min/max:  {ci_qwen_avg[-2]:.4f} - {ci_qwen_avg[-1]:.4f}")
    print(f"Context (Qwen) min/max CI:  {ci_qwen_minmax[0]:.4f} - {ci_qwen_minmax[1]:.4f} (mean={ci_qwen_minmax[2]:.4f})")
    print(f"Context (Qwen) min/max min/max:  {ci_qwen_minmax[-2]:.4f} - {ci_qwen_minmax[-1]:.4f}")

    print(f"Static (GloVe) avg CI:  {ci_glove_avg[0]:.4f} - {ci_glove_avg[1]:.4f} (mean={ci_glove_avg[2]:.4f})")
    print(f"Static (GloVe) avg min/max:  {ci_glove_avg[-2]:.4f} - {ci_glove_avg[-1]:.4f}")
    
    print(f"Static (GloVe) min/max CI:  {ci_glove_minmax[0]:.4f} - {ci_glove_minmax[1]:.4f} (mean={ci_glove_minmax[2]:.4f})")
    print(f"Static (GloVe) min/max min/max:  {ci_glove_minmax[-2]:.4f} - {ci_glove_minmax[-1]:.4f}")

if __name__ == "__main__":
    main()
