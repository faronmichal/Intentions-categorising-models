## Model Comparison

In this project, I tested several approaches to classifying customer messages.  
I started with a basic LSTM model, then added FastText embeddings.  
I also built a more advanced version (two-layer, bidirectional),  
and finally tested a transformer-based language model trained with Hugging Face.

| Model | Description / Variant | Test Accuracy |
|--------|------------------------|---------------|
| LSTM | Basic LSTM model | 86.05% |
| LSTM + FastText | LSTM with FastText embeddings | 88.08% |
| LSTM (advanced) | Two-layer, bidirectional version | 85.91% |
| LLM (transformer) | Language model trained with Hugging Face Trainer | 95.88% |

The transformer model achieved the best results and clearly outperformed the LSTM variants