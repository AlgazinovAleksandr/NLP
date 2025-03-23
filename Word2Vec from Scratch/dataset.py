import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

class SkipGramDataset(Dataset):
    def __init__(self, file_path, window_size, min_count=5):
        with open(file_path, 'r') as f:
            documents = [line.strip().split() for line in f]
        
        word_counts = Counter(word for doc in documents for word in doc)
        vocab = [word for word, count in word_counts.items() if count >= min_count]
        self.vocab_size = len(vocab)
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        # Compute negative sampling probabilities (frequency^(3/4) normalized)
        counts = np.array([word_counts[word] for word in vocab])
        probs = counts ** (3/4) # make it more likely to sample less frequent words as negatives, which improves the quality of the embeddings (please refer to the original paper for more details)
        probs /= probs.sum() # make sure probabilities add up to 1
        self.negative_sampling_probs = torch.from_numpy(probs).float()
        
        # Generate positive (target, context) pairs
        self.pairs = []
        for doc in documents:
            doc_indices = [self.word_to_index[word] for word in doc if word in self.word_to_index]
            for i in range(len(doc_indices)):
                target = doc_indices[i]
                for j in range(max(0, i - window_size), min(len(doc_indices), i + window_size + 1)):
                    if j != i:
                        context = doc_indices[j]
                        self.pairs.append((target, context))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def convert_word_to_idx(self, word):
        return self.word_to_index[word]
    
    def convert_idx_to_word(self, idx):
        return self.index_to_word[idx]