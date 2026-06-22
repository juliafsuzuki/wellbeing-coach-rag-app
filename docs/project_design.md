# Design Decisions

Full architecture, chunking, retrieval, and evaluation decisions for the DanceSport Wellbeing Coach RAG app (LangChain + Pinecone + OpenAI + Streamlit).

## Stack

- **Orchestration:** LangChain + LangGraph + LangSmith
- **LLM:** OpenAI `gpt-4.1-mini` — routing & judge (temp=0), generation (temp=0.1)
- **Embeddings:** OpenAI `text-embedding-3-small` (dim=1536)
- **Vector store:** Pinecone Serverless, cosine similarity
- **UI:** Streamlit (`app.py`)
- **Notebook:** `1_wellbeing_coach_rag_app_langchain.ipynb`
- **Observability:** LangSmith project `wellbeing-coach-rag-app-langchain`

## Corpus

Single PDF: *Dance To Your Maximum* by Maximiliaan Winkelhuis.
Place the file at `data/e-Book_dance-to-your-maximum.pdf` before running the notebook.

Structure: workbook with Part One (Competition Day), Part Two (Season), Part Three (Career).
Numbered chapters (e.g. 1-2, 1-4, 2-3, 2-8, 3-1), personal tests, evaluation forms, questionnaires, appendices.

## Ingestion

The PDF is fully image-based (scanned; no text layer). Standard PDF loaders return blank pages.

- **PyMuPDF (`fitz`)** renders each page to a grayscale image at 150 DPI
- **Tesseract (`pytesseract`)** extracts text via OCR
- Results are cached to `data/ocr_cache.json` — OCR runs only once
- Detect structure via regex: Part headings, Chapter headings, section_type markers
- Clean OCR conservatively: merge hyphenated line-breaks, normalise whitespace, collapse blank lines

## Chunking (hierarchical)

- Prose chapters: `chunk_size=1200`, `chunk_overlap=200`
- Tests / forms / appendices: `chunk_size=2200`, `chunk_overlap=150`
- `min_chunk_size=100` — chunks below this are discarded
- `RecursiveCharacterTextSplitter` per section type

## Metadata per chunk

| Field | Values |
|---|---|
| `source` | `dance_to_your_maximum` |
| `part` | `Part One` · `Part Two` · `Part Three` |
| `part_scope` | `competition_day` · `season` · `career` |
| `chapter` | e.g. `1-2` |
| `chapter_title` | free text |
| `section_type` | `prose` · `test` · `form` · `appendix` · `intro` · `toc` |
| `page_start` | integer |
| `page_end` | integer |
| `chunk_id` | MD5 hash (12 chars) |
| `parent_id` | `page_{page_start}` |

## Retrieval

- Pure dense retrieval (V1); no BM25 hybrid or cross-encoder re-ranking
- `top_k=6` default (unfiltered); `top_k=4` when a route filter is applied
- Metadata filters: `part_scope` and/or `section_type`
- Fallback to unfiltered `top_k=6` if filtered search returns 0 results
- Refusal path when `documents` list is empty after retrieval

## Query routing (LangGraph)

| Question intent | `part_scope` | `section_type` |
|---|---|---|
| Competition day, morning routine, warm-up | `competition_day` | `prose` |
| Season planning, choreography, exercises | `season` | `prose` |
| Long-term goals, career arc, identity | `career` | `prose` |
| Questionnaire, self-assessment | (any) | `test` |
| Evaluation form, worksheet | (any) | `form` |
| Appendix content | (any) | `appendix` |
| Unclear or cross-part | `general` | `general` |

## Citations

Format: `[Dance To Your Maximum, Chapter 1-2, Page 21–24]`  
Inline in answer body after each claim. Drawn from chunk metadata — never fabricated.

## Evaluation

- 15-question fixed test set (all 3 parts, both prose and form types)
- LLM judge: `gpt-4.1-mini`, temp=0
- Target: ≥ 90% faithfulness (claims fully supported by retrieved context)
- Reproducible in-notebook evaluation pipeline (Section 11)
- Manual spot-check on a representative subset

## Environment

- `.env` file: populate manually with `OPENAI_API_KEY`, `PINECONE_API_KEY`, `LANGCHAIN_API_KEY`
- Virtual environment: `.venv` (reconstruct with `pip install -r requirements.txt`)
- LangSmith tracing: disabled by default (`LANGCHAIN_TRACING_V2=false`); enable by setting to `true` with a valid `LANGCHAIN_API_KEY`
