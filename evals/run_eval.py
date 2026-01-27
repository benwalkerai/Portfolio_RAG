# Import Modules
import json
import sys
import os


# Add root folder to python path so we can import 'rag'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.store import get_vector_store

def run_eval():
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k":4})

    with open("evals/dataset.json", "r") as f:
        dataset = json.load(f)
    print(f"\n Running Eval on {len(dataset)} questions...\n")

    score = 0

    for item in dataset:
        q = item["question"]
        target = item["expected_phrase"]

        docs = retriever.invoke(q)
        retrieved_text = " ".join([d.page_content for d in docs]).lower()

        found = target.lower() in retrieved_text
        marker = "✅" if found else "❌"

        if found:
            score += 1

        print(f"{marker} Q: {q}")
        if not found:
            print(f"  Expected: '{target}'")
            print(f"  Got (snippets): {retrieved_text[:100]}...")
        
        final_score = (score / len(dataset)) * 100
        print(f"\n Final Recall@4 Score: {final_score:.1f}%")

if __name__ == "__main__":
    run_eval()