"""
Main entry point.

Usage:
    1. Put 2 requirement PDFs in the `data/` folder.
    2. Make sure Ollama is running and you have pulled the embedding model:
         ollama pull nomic-embed-text
       (phi3:mini should already be installed per your setup)
    3. Run:
         python main.py --build       # build the vector store (do this once, or after adding new PDFs)
         python main.py --query "..."  # ask for test cases on a specific requirement
"""

import argparse
import json
from pdf_ingest import load_and_chunk_all_pdfs
from vectorstore import build_vectorstore, load_vectorstore
from generate import generate_test_cases, print_test_cases, NotARequirementError


def build():
    chunks = load_and_chunk_all_pdfs()
    if not chunks:
        return
    build_vectorstore(chunks)


def query(requirement: str, save_path: str = None):
    collection = load_vectorstore()

    try:
        test_cases, context_chunks = generate_test_cases(collection, requirement)
    except NotARequirementError as e:
        print(f"\n{e}")
        return

    print("\n--- Retrieved context used ---")
    for c in context_chunks:
        print(f"  - {c['source']} | {c['section']}")

    print("\n--- Generated Test Cases ---")
    print_test_cases(test_cases)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(test_cases, f, indent=2)
        print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Extract PDFs and (re)build the vector store")
    parser.add_argument("--query", type=str, help="Requirement/user story to generate test cases for")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save output JSON")
    args = parser.parse_args()

    if args.build:
        build()
    elif args.query:
        query(args.query, args.save)
    else:
        print("Use --build first, then --query \"your requirement text\"")