# NLP Project: Exploring Modern NLP Techniques

This project delves into various aspects of Natural Language Processing, covering prompt engineering, architectural analysis of transformer-based models, and hands-on model modification.

## 1. Prompt Engineering

### 1.1 Unique Prompt Design & Testing 
* Designed a unique prompt to test the capabilities of two different Large Language Models (LLMs)
* The prompt targeted a specific task with a single, verifiable answer, such that one selected model consistently succeeded while the other failed
* Tested models: Gemini 2.5 Pro and GPT 4o reasoninng models
* Consistency was verified by running the prompt three times on each model
* *(Results, screenshots, and analysis are detailed in the accompanying report)

### 1.2 In-Context Learning for Reasoning 
* Designed prompts to solve reasoning problems from the GSM-8K dataset using in-context learning
* Implemented both few-shot (with demonstrations) and zero-shot prompts
* Ensured model outputs were structured (e.g., JSON/YAML) including Question ID, Reasoning Process, Final Answer, and Difficulty Classification
* *(Comparison of few-shot vs. zero-shot performance is included in the report)

## 2. Architecture Analysis

* Calculated and compared the Model Size, KV Cache Size, and Forward Pass FLOPs for different transformer-based architectures
* Analyzed architectures included:
    * Standard GPT (Multi-Head Attention)
    * GPT with Grouped Query Attention (GQA) 
    * Mamba-2 
    * Gated Linear Attention (GLA)
* Calculations were based on provided notations and assumptions (e.g., bfloat16 precision, tied embeddings, negligible FLOPs for certain ops)
* Key metrics derived:
    * **Model Size:** Total trainable parameters (embeddings + transformer layers)
    * **KV Cache Size:** Memory required for storing past keys/values during autoregressive generation
    * **FLOPs:** Floating-point operations for a forward pass, approximating computational cost
* *(Detailed derivations, results table, and comparative analysis of computational/memory trade-offs are provided in the report)

## 3. Model Architecture Modification

This part involved modifying the Qwen2.5 model architecture. The specific modifications implemented in `modeling_qwen2.py` include:

* **Learned Position Encoding:** Replaced RoPE with learned absolute position embeddings added after the input embedding layer
* **Dynamic Tanh (DyT):** Replaced all RMSNorm normalization layers with the DyT module
* **ReLU-Attention:** Replaced the standard Softmax in the attention mechanism with ReLU-Attention ($a_{ij} = ReLU(q_i k_j^T) / L$)
* **Layer-Dependent Attention Masking:** Implemented a varying attention mask strategy across layers:
    * First layer: Causal mask
    * Middle layers: Custom block-based mask pattern
    * Last layer: Dilated sliding window attention (window=128, dilation=2)
