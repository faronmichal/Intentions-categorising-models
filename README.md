---

# Intentions Categorising Models

A repository of machine learning models to **classify user/customer intentions from text** - including LSTM networks, FastText models, and transformer-based language models.

---

## Project Overview

Understanding user intent from text is essential in customer support systems, chatbots, and other NLP-driven applications. This project explores multiple modelling approaches - from classical word-vector-based methods to modern transformers - to evaluate performance trade-offs in intention classification tasks.

---

## Model Comparison

| Model Variant                  | Description                                 | Test Accuracy |
| ------------------------------ | ------------------------------------------- | ------------- |
| **LSTM (Basic)**               | Single-layer LSTM                           | **86.05%**    |
| **LSTM + FastText Embeddings** | LSTM with FastText pretrained vectors       | **88.08%**    |
| **LSTM (Advanced)**            | Two-layer, bidirectional LSTM               | **85.91%**    |
| **FastText (Native)**          | Facebook’s supervised FastText classifier   | **91.07%**    |
| **Transformer LLM**            | Fine-tuned transformer model (Hugging Face) | **95.88%**    |

> Both FastText and transformer models significantly outperform LSTM architectures, with transformers achieving the highest accuracy overall.

---

## Architecture & Approach

1. **LSTM-based Models**

   * A baseline single-layer LSTM.
   * A version using **FastText embeddings**, improving performance through richer subword-aware vectors.
   * A deeper, bidirectional LSTM model capturing sequential context in both directions.

2. **FastText Model**

   * Uses **native FastText supervised training** (`fasttext` library).
   * Learns embeddings and classifier jointly.
   * Efficient, fast, and robust thanks to subword n-gram representations.
   * Achieves strong performance while remaining extremely lightweight.

3. **Transformer Model**

   * A pre-trained transformer fine-tuned on the intentions dataset (via Hugging Face Trainer API).
   * Learns contextual semantic representations.
   * Produces the highest accuracy across all evaluated models.

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/faronmichal/Intentions-categorising-models.git
   cd Intentions-categorising-models
   ```

2. Install dependencies (example):

   ```bash
   pip install -r requirements.txt
   ```

3. Train a model:

   * Use the appropriate script or notebook (`lstm_networks.py` or the LLM fine-tuning notebook).
   * Prepare your dataset.
   * Adjust hyperparameters as needed.

4. Evaluate & predict:

   * Use built-in evaluation routines.
   * Save and load trained models for inference.

---

## Repository Contents

* `lstm_networks.py` - LSTM and fasttext architectures and training utilities
* `polish_llm_fine_tuning.py` - Transformer fine-tuning workflow for Polish intentions

---

## Key Insights

* **FastText provides an excellent baseline**, outperforming all LSTM models while offering fast training and inference.
* **LSTM performance depends heavily on embedding quality** - pretrained FastText embeddings improve results noticeably.
* **Transformers deliver the best performance**, confirming that contextual representations outperform classical architectures in intent classification.
* **Model choice is task-dependent** - FastText is ideal for lightweight, fast systems; transformers excel where accuracy is the priority.

---

