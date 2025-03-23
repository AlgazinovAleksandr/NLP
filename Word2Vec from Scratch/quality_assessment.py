import torch
import os
from skipgram import SkipGramModel
import torch.nn.functional as F
from collections import Counter
from dataset import SkipGramDataset

import random
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Ensure Bokeh output is set to notebook if using Jupyter
output_notebook()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'cut_text.txt'  # Decided to use the non-lemmatized text
min_count = 20
window_size = 2
dataset = SkipGramDataset(file_path, window_size=window_size, min_count=min_count)
dataset.vocab_size


min_count = 20
with open(file_path, 'r') as f:
      documents = [line.strip().split() for line in f]
      word_counts = Counter(word for doc in documents for word in doc)
      vocab = [word for word, count in word_counts.items() if count >= min_count]

def most_similar(model_name, word, top_n=5, device=device):
    """
    Find the top_n most similar words to the given word based on cosine similarity.
    """
    # Load the checkpoint
    save_path = os.path.join('model_checkpoints', model_name)
    checkpoint = torch.load(save_path, map_location=device)

    embedding_dim = 100
    vocab_size = len(vocab)
    idx_to_word = dataset.convert_idx_to_word
    word_to_idx = dataset.word_to_index

    # Initialize and load the model
    model = SkipGramModel(vocab_size, embedding_dim=100).to(device)
    model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode

    # Get and normalize target embeddings
    embeddings = model.target_embeddings.weight
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    # Check if the word is in the vocabulary
    if word not in word_to_idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    # Get the embedding of the input word
    idx = word_to_idx[word]
    word_emb = normalized_embeddings[idx]

    # Compute cosine similarities (dot product since embeddings are normalized)
    similarities = torch.matmul(normalized_embeddings, word_emb)

    # Exclude the input word by setting its similarity to -1
    similarities[idx] = -1

    # Get the top_n most similar words
    _, top_indices = torch.topk(similarities, k=top_n, largest=True)
    similar_words = [idx_to_word(i.item()) for i in top_indices]

    return similar_words

def visualize_embeddings_interactive(model_name, num_words=500, seed=42, device='cpu',
                                    radius=10, alpha=0.5, color='blue',
                                    width=800, height=800, show=True):
    """
    Draws an interactive plot for word embeddings with hover tooltips showing the words.
    Returns:
        bokeh.plotting.Figure: The Bokeh figure object for further customization.

    inspired by: https://github.com/ashaba1in/hse-nlp/blob/main/2023/seminars/week1_word_embeddings.ipynb
    (NLP course from my undergrad)
    """
    # Load the model checkpoint
    save_path = os.path.join('model_checkpoints', model_name)
    checkpoint = torch.load(save_path, map_location=device)

    embedding_dim = 100
    vocab_size = len(vocab)
    idx_to_word = dataset.convert_idx_to_word
    word_to_idx = dataset.word_to_index

    # Initialize and load the model
    model = SkipGramModel(vocab_size, embedding_dim=100).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Get target embeddings
    embeddings = model.target_embeddings.weight

    # Set random seed and sample indices
    random.seed(seed)
    sampled_indices = random.sample(range(vocab_size), num_words)

    # Extract embeddings for sampled words
    sampled_embeddings = embeddings[sampled_indices].cpu().detach().numpy()

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=seed)
    embeddings_2d = tsne.fit_transform(sampled_embeddings)

    # Get corresponding words
    sampled_words = [idx_to_word(idx) for idx in sampled_indices]

    # Prepare data for Bokeh
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    if isinstance(color, str):
        color = [color] * len(x)  # Ensure color matches the number of points

    # Create ColumnDataSource with x, y, color, and word data
    data_source = bm.ColumnDataSource({
        'x': x,
        'y': y,
        'color': color,
        'word': sampled_words
    })

    # Create Bokeh figure with interactive tools
    fig = pl.figure(
        title="Interactive t-SNE Visualization of Word Embeddings",
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll='wheel_zoom'
    )

    # Add scatter plot
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    # Add hover tooltips to display the word
    fig.add_tools(bm.HoverTool(tooltips=[("word", "@word")]))

    # Display the plot if requested
    if show:
        pl.show(fig)

    return fig