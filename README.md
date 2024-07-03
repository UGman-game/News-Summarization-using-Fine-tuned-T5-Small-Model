# News-Summarization-using-Fine-tuned-T5-Small-Model

This project involves fine-tuning a pre-trained T5 model for the task of text summarization. Using the T5-small architecture, the model was trained on a dataset of articles and their respective highlights to generate concise summaries. The key steps in this project include:

**Data Loading and Preprocessing:**

+ Loaded and analyzed a CSV dataset of articles and highlights.
+ Preprocessed the data by tokenizing and padding/truncating the articles and highlights using the T5 tokenizer.

**Model Training:**

+ Fine-tuned the T5 model on the preprocessed data using a custom DataLoader.
+ Trained the model over multiple epochs, optimizing it with the AdamW optimizer.
+ Calculated loss and perplexity to monitor training performance.

**Inference:**

+ Saved the fine-tuned model.
+ Implemented a mechanism for summarizing new input text using the fine-tuned model.

**Technical Stack:**

+ Libraries: PyTorch, Transformers, Pandas
+ Framework: Google Colab
+ Model: T5-small
+ Data Handling: DataLoader, TensorDataset

This project demonstrates the ability to apply state-of-the-art NLP techniques for text summarization, including data preprocessing, model fine-tuning, and generating summaries using a pre-trained transformer model.

## Key Features:

+ Fine-tuned T5 Model: Customized for generating summaries of given articles.
+ Data Handling: Efficient preprocessing and loading of large datasets.
+ Training Metrics: Monitoring of loss and perplexity to evaluate model performance.
+ Inference Capability: Real-time summarization of new text inputs.

## Applications:

+ Automated summarization of lengthy documents.
+ Enhancing content creation workflows.
+ Improving information retrieval systems.
