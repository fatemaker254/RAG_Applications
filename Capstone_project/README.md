# QA Test Case Generation Agent (Local RAG with Ollama + phi3:mini)

## 1. Setup

```bash
pip install -r requirements.txt
```

Make sure Ollama is running locally, and pull the embedding model (phi3:mini is already installed per your setup, you only need this one extra):

```bash
ollama pull nomic-embed-text
```

## 2. Add your data

Drop your 2 requirement PDFs into the `data/` folder:

```
qa_test_case_agent/
  data/
    requirement_doc_1.pdf
    requirement_doc_2.pdf
```

## 3. Build the vector store (run once, or whenever you add/change PDFs)

```bash
python main.py --build
```

This will:
- Extract text from each PDF
- Chunk it by section headers (or fixed-size if no headers are found)
- Embed each chunk using `nomic-embed-text`
- Store everything in a local Chroma DB (`chroma_db/` folder)

## 4. Generate test cases

```bash
python main.py --query "Citizen registers a complaint through the Registration module"
```

Optional: save output to a file

```bash
python main.py --query "Advanced search must return results within 10-15 seconds" --save output.json
```

## How it works (pipeline)

```
PDFs -> extract text -> chunk by section -> embed (nomic-embed-text) -> Chroma vector store
                                                                              |
User requirement query -> embed query -> retrieve top-k relevant chunks ----+
                                                                              |
                          Prompt (context + requirement) -> phi3:mini -> JSON test cases
```

## Notes / things you may need to tune

- **phi3:mini and JSON formatting**: small models occasionally return malformed JSON.
  `generate.py` retries automatically and strips markdown fences. If it still fails often,
  try lowering `temperature` further in `generate.py`, or add one example test case
  directly in the prompt (few-shot) to steady the format.
- **Chunking**: `pdf_ingest.py` tries to split by section headers (e.g. "1. INTRODUCTION",
  "A1 REGISTRATION"). If your PDFs don't have this style of heading, it automatically
  falls back to fixed-size chunking — check the console output to see which mode was used.
- **TOP_K**: in `config.py`, controls how many chunks are retrieved per query. Increase if
  your requirements docs are large and generation feels like it's missing context.
