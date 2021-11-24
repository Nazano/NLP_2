""" Testez notre modèle de QA avec ce script
"""

import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import sentence_transformers

def TopK(x, k):
    a = dict([(i, j) for i, j in enumerate(x)])
    sorted_a = dict(sorted(a.items(), key = lambda kv:kv[1], reverse=True))
    indices = list(sorted_a.keys())[:k]
    values = list(sorted_a.values())[:k]
    return (values, indices)

def answer_topk(model, question, top_k=5):
    top_k = min(top_k, len(dataset['context']))

    query_embedding = model.encode(question, convert_to_numpy=True)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = sentence_transformers.util.cos_sim(query_embedding, contexts_embeddings)[0]
    top_results = TopK(cos_scores, top_k)  # np.argpartition(cos_scores, -top_k)[-top_k:]

    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Confidence: {score}\n{dataset['context'].iloc[idx]}")
        return



if __name__ == "__main__":
    with open("pickle/dataset", "rb") as f:
        dataset = pickle.load(f)

    with open("pickle/contexts_embeddings", "rb") as f:
        contexts_embeddings = pickle.load(f)

    model = SentenceTransformer('msmarco-roberta-base-v3')

    print("QA me ✨")
    while True:
        question = input("?> ")
        answer_topk(model, question)

