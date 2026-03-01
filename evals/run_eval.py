import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.store import get_vector_store

def run_eval():
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    with open("evals/dataset.json", "r") as f:
        dataset = json.load(f)
    
    print(f"\nRunning Eval on {len(dataset)} questions...\n")

    score = 0

    for item in dataset:
        question = item["question"]
        expected_phrase = item["expected_phrase"]

        docs = retriever.invoke(question)
        retrieved_text = " ".join([d.page_content for d in docs]).lower()

        found = expected_phrase.lower() in retrieved_text
        marker = "✅" if found else "❌"

        if found:
            score += 1

        print(f"{marker} Q: {question}")
        if not found:
            print(f"  Expected: '{expected_phrase}'")
            print(f"  Got (snippets): {retrieved_text[:100]}...")
        
    final_score = (score / len(dataset)) * 100
    print(f"\nFinal Recall@4 Score: {final_score:.1f}%")

if __name__ == "__main__":
    run_eval()
