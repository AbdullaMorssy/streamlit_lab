import numpy as np

# Sample text documents
DOCUMENTS = [
    "Large Language Models (LLMs) enable advanced text generation.",
    "Transformers use self-attention for better NLP performance.",
    "Fine-tuning LLMs improves accuracy for specific domains.",
    "Ethical AI involves fairness, transparency, and accountability.",
    "Zero-shot learning allows LLMs to handle unseen tasks.",
    "Embedding techniques convert words into numerical vectors.",
    "LLMs can assist in chatbots, writing, and summarization.",
    "Evaluation metrics like BLEU score assess text quality.",
    "The future of AI includes multimodal models integrating text and images.",
    "Mistral AI optimizes LLM performance for efficiency.",
]


def main():
    # Save documents to a text file
    with open("documents.txt", "w", encoding="utf-8") as f:
        for doc in DOCUMENTS:
            f.write(doc + "\n")

    # Generate random embeddings for demonstration
    embedding_dim = 512
    num_documents = len(DOCUMENTS)
    document_embeddings = np.random.rand(num_documents, embedding_dim).astype(np.float32)
    np.save("embeddings.npy", document_embeddings)


if __name__ == "__main__":
    main()
