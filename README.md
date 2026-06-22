# DanceSport Wellbeing RAG Application

## Problem Statement

DanceSport athletes navigate a complex mix of physical, mental, emotional, psychological, and strategic challenges while preparing for showcase performances and competitions. While expert coaching and valuable educational resources exist, access to them is often limited and expensive. Many insights remain buried in books, articles, and other written materials, making discovery largely a matter of chance.

Athletes need timely access to trusted guidance when they need it most. To support effective training and performance, coaching knowledge should be readily accessible through a conversational, queryable experience.

## Solution Overview

The DanceSport Wellbeing Coach is a single-source Retrieval-Augmented Generation (RAG) application that enables DanceSport athletes to access trusted guidance on training, performance, and wellbeing by asking natural-language questions related to showcase and competition preparation.

The coach is designed to serve multiple roles, including a/an:
- Evidence-based advisor that provides trusted guidance specially for showcase and competition success.
- anceSport coach that helps athletes train effectively, perform with confidence, and compete at their highest potential.
- Full Potential coach that empowers athletes to unlock their maximum potential both on and off the dance floor.
- Wellbeing and performance companion that helps athletes strengthen the physical, mental, emotional, and performance dimensions.
- Personalized companion that supports athletes in their pursuit of excellence, continuous improvement, and mastery.

Athletes can ask questions and receive guidance across six categories:
1. Performance readiness
2. Practice and preparation
3. Musicality and timing
4. Confidence and stage presence
5. Expression and storytelling
6. Mindset and mental performance

All responses are grounded exclusively in an PDF e-book, "Dance To Your Maximum" by Maximiliaan Winkelhuis and include inline citations to the relevant chapter and page, enabling athletes to verify the source material and explore the concepts in greater depth.

### Corpus
A single corpus is used to build the solution. A corpus is a PDF e-book, "Dance To Your Maximum" by Maximiliaan Winkelhuis.

It is a structured workbook with:
- Part One: The Competition Day
- Part Two: The Season
- Part Three: The Dancer’s Career
- Numbered chapters such as 1-2 Stress and preparations, 1-4 The Nine Step Connection Model, 2-3 Planning your choreography, 2-8 Exercises during the season, and 3-1 Career planning
- Workbook sections including personal tests, questionnaires, evaluation forms, and appendices
  
## Repository layout

```
wellbeing-coach-rag-app/
├── app.py                          # Streamlit chat UI
├── rag_chain.py                    # RAG graph, router, retriever, generator
├── 1_wellbeing_coach__rag_app_langchain.ipynb  # End-to-end pipeline + evaluation
├── data/
│   ├── e-Book_dance-to-your-maximum.pdf        # Source corpus (316 pages, image-based)
│   └── ocr_cache.json                          # OCR output cache (run-once)
├── images/
│   └── home_page.jpg                           # Home page screenshot
├── docs/
│   ├── specification.md                        # Functional & technical specification
│   ├── project_design.md                       # Architecture & design decisions
│   ├── MEMORY.md                               # Memory index
│   └── reference_github.md                     # Repo reference
├── generate_diagrams.py            # Diagram generation utility
└── .venv/                          # Python virtual environment
```

### Documentation

- [Project Specification](docs/specification.md): Both functional & technical: user stories, UI categories, RAG graph, routing, metadata schema, evaluation, constraints.
- [Project Design](docs/project_design.md): Architecture, chunking, retrieval, and evaluation decisions.

<br />

# Technical Overview

<img width="1672" height="941" alt="6-EndToEndWorkflow" src="https://github.com/user-attachments/assets/d0e37f8d-ef2b-4fdd-99c3-f166a12b1edb" />

#### START

### Pipeline Overview

```
Phase 1 — Ingest
PDF → PyMuPDF (renders each page to grayscale image)
    → Tesseract OCR (extracts text)
    → JSON disk cache (data/ocr_cache.json)
    → Hierarchical chunking (RecursiveCharacterTextSplitter)
    → OpenAIEmbeddings (text-embedding-3-small)
    → Pinecone index

Phase 2 — RAG Pipeline (LangGraph, 3 nodes)
User query → route node (classify part_scope + section_type)
           → retrieve node (Pinecone similarity search)
           → generate node (LLM answer with citations)

Phase 3 — UI
Streamlit (app.py) → renders categorised example questions + chat interface
```

---

### 0. Getting Started

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment (edit manually — never commit)
# .env must contain:
OPENAI_API_KEY=...
PINECONE_API_KEY=...
LANGCHAIN_API_KEY=          # optional; leave blank to disable LangSmith tracing
LANGCHAIN_TRACING_V2=false  # set true only when LANGCHAIN_API_KEY is populated
LANGCHAIN_PROJECT=wellbeing-coach-rag-app-langchain

# Ingest PDF and build Pinecone index
jupyter notebook 1_wellbeing_coach_rag_app_langchain.ipynb

# Launch chatbot UI
streamlit run app.py
```

---

### 1. PDF Ingestion via OCR

**Issue encountered:** PyPDFLoader returned empty text for every page.

**Root cause:** The source PDF (*Dance To Your Maximum*, Maximiliaan Winkelhuis) was produced with Adobe Acrobat 6.0 (2008) as a scanned image — each page is a photograph of text, not a selectable text layer. `PyPDFLoader` and `pypdf` can only read embedded text; they return nothing for image-based PDFs.

**Solution:**

1. PyMuPDF (`fitz`) renders each page to a grayscale image at 300 DPI.
2. Tesseract (`pytesseract`) extracts text from the image via OCR.
3. Results are cached to `data/ocr_cache.json` so OCR runs only once across sessions.

```
data/e-Book_dance-to-your-maximum.pdf  →  PyMuPDF  →  Tesseract  →  data/ocr_cache.json
```

---

### 2. Chunking Strategy

**Method:** Hierarchical, structure-aware chunking using `RecursiveCharacterTextSplitter`.

| Section type | chunk_size | chunk_overlap | min_chunk_size |
|---|---|---|---|
| Prose chapters | 1 200 chars | 200 chars | 100 chars |
| Tests / forms / appendices | 2 200 chars | 150 chars | 100 chars |

Chunks below `min_chunk_size` are discarded.

**Metadata stored per chunk:**

| Field | Example values |
|---|---|
| `source` | `dance_to_your_maximum` |
| `part` | `Part One` |
| `part_scope` | `competition_day` · `season` · `career` |
| `chapter` | `1-2` · `2-8` · `3-1` |
| `chapter_title` | free text |
| `section_type` | `prose` · `test` · `form` · `appendix` |
| `page_start` | integer |
| `page_end` | integer |
| `chunk_id` | UUID |
| `parent_id` | UUID |

---

### 3. Vector Database — Pinecone

| Setting | Value |
|---|---|
| Index name | `wellbeing-coach-rag` |
| Cloud / region | AWS `us-east-1` (serverless) |
| Similarity metric | Cosine |
| Embedding model | `text-embedding-3-small` |
| Vector dimension | 1 536 |

Metadata filters used at query time: `part_scope` and/or `section_type`.

---

### 4. Query Routing (LangGraph)

A 3-node stateful graph (`route → retrieve → generate`) processes every query.

**Route node** classifies the query into two dimensions using `gpt-4.1-mini` (temp=0):

| Dimension | Options |
|---|---|
| `part_scope` | `competition_day` · `season` · `career` · `general` |
| `section_type` | `prose` · `test` · `form` · `appendix` · `general` |

**Retrieve node** uses the route to filter Pinecone results:

- With a filter match → `top_k = 4` (metadata-filtered search)
- No filter (general query) → `top_k = 6` (unfiltered)
- Fallback to unfiltered `top_k = 6` if filtered search returns 0 results

**Generate node** formats retrieved chunks as context, appends source citations, and calls `gpt-4.1-mini` (temp=0.1) to produce the final answer.

---

### 5. Chatbot UI — Streamlit

**File:** `app.py`

- Page icon: 💃, layout: centered
- 6 categorised question panels in a 2-column grid (always visible, above chat history)
- Questions trigger via `st.session_state.pending_question` to avoid conflicts with `st.chat_input`
- Chat history persisted in `st.session_state.messages` for the session lifetime

**Categories:** Performance Readiness · Practice & Preparation · Musicality & Timing · Confidence & Stage Presence · Expression & Storytelling · Mindset & Mental Performance

---

### 6. Tagging & Citation

Every claim in the generated answer must carry an explicit tag and an inline source citation.

**Claim tags:**

| Tag | Meaning |
|---|---|
| `[KNOWN]` | Established training fact |
| `[COMPUTED]` | Calculated value |
| `[INFERRED]` | Logical deduction from context |
| `[COMMON]` | Standard domain knowledge |
| `[FRAME]` | Symbolic system (coherent ≠ real-world claim) |
| `[GUESS]` | No supporting basis |

**Citation format** (inline, after each claim):

```
[Dance To Your Maximum, Chapter 1-2, pp. 21–24]
```

Chapter numbers follow the book's own numbering scheme (e.g. `1-2`, `2-8`, `3-1`). Page range uses `pp.` for multi-page spans and `p.` for a single page. Citations are drawn from chunk metadata (`chapter`, `page_start`, `page_end`) — never fabricated.

---

### 7. Evaluation

**Method:** Automated LLM-as-judge using `gpt-4.1-mini`, run inside the notebook.

| Setting | Value |
|---|---|
| Test set | 15 fixed questions |
| Primary metric | Faithfulness — % of claims fully supported by retrieved context |
| Target | ≥ 90% faithful answers |
| Secondary check | Manual spot-check on a representative subset |

The evaluation pipeline is reproducible and self-contained in the notebook (Section 11).

---

### 8. Observability — LangSmith

| Setting | Value |
|---|---|
| Project name | `wellbeing-coach-rag-app-langchain` |
| Status | **Disabled** (`LANGCHAIN_TRACING_V2=false`) |
| Activation | Set `LANGCHAIN_TRACING_V2=true` and populate `LANGCHAIN_API_KEY` in `.env` |

---

### AI Stack

| Layer | Tool / Model |
|---|---|
| Orchestration | LangChain + LangGraph (stateful graph) + LangSmith (observability) |
| LLM — routing & judge | `gpt-4.1-mini`, temperature=0 |
| LLM — generation | `gpt-4.1-mini`, temperature=0.1 |
| Embeddings | OpenAI `text-embedding-3-small` (dim=1 536) |
| Vector store | Pinecone Serverless — index `wellbeing-coach-rag`, cosine, AWS us-east-1 |
| PDF rendering | PyMuPDF (`fitz`) |
| OCR | Tesseract (`pytesseract`) |
| UI | Streamlit (`app.py`) |
| Notebook | `1_wellbeing_coach_rag_app_langchain.ipynb` |

#### END

## Chatbot UI

The home page surfaces 6 pre-built categories with clickable example questions, arranged in a two-column grid.

<img width="378" height="692" alt="Screenshot 2026-06-21 222442" src="https://github.com/user-attachments/assets/819e36f8-1ae6-4d46-ae61-886fcf705d5b" />

<br />

### Lists of questions for 6 categories:

**🎯 Performance Readiness**
- How do I know if I am ready to perform my showcase?|
- What should I focus on during the final weeks before my showcase?
- How can I reduce mistakes during my performance?
- What should I do if I forget part of my routine on the floor?

**📋 Practice & Preparation**
- What is the best way to practice my showcase routine?
- How often should I practice between lessons?
- How do I remember my showcase routine more effectively?
- How can I make my practice sessions more productive?
- How do I transition from learning choreography to performing it?

**🎵 Musicality & Timing**
- How can I improve my timing with the music?
- How do I stay on time when I get nervous?
- How can I better connect my movements to the music?
- How important is breathing for timing, movement, and performance quality?

**✨ Confidence & Stage Presence**
- How can I look more confident on the dance floor?
- How do I manage performance anxiety or stage fright?
- How can I project confidence even when I feel nervous?
- How can I make a strong first impression when I enter the floor?

**💃 Expression & Storytelling**
- How can I make my performance more expressive?
- How can I better tell the story of the dance?
- How do I connect emotionally with the music and my audience?
- What makes a showcase performance memorable?

**🧠 Mindset & Mental Performance**
- Why is mental clarity important in DanceSport?
- How can I stay mentally focused during my performance?
- How do I recover mentally after making a mistake?
- How can visualization improve my showcase performance?

Users can also type a free-text question in the persistent chat input at the bottom.

### Answer format:

Every claim is tagged and cited:

- **Epistemic tags:** `[KNOWN]`, `[INFERRED]`, `[COMPUTED]`, `[COMMON]`, `[FRAME]`, `[GUESS]`
- **Confidence:** HIGH (≥80%) · MED (50–80%) · LOW (20–50%) · VERY LOW (<20%) · UNKNOWN
- **Citation:** `[Dance To Your Maximum, Chapter X-Y, pp. Z–W]`
- **Refusal:** When no relevant context is retrieved, the answer begins with `"I don't have that in my knowledge base."` — no fabrication.

<br />

## Scope and Constraints

- Single corpus (no multi-document upload): all knowledge is derived from an e-book, Dance To Your Maximum. 
- No cross-session memory (chat history clears on reload)
- No re-ranking, no streaming
- Local `.env` for secrets; single-process Streamlit deployment

## Key capabilities

- Conversational Q&A via a Streamlit web interface, accessible to non-technical users
- Intelligent query routing (powered by LangGraph) that directs questions to the most relevant section of the knowledge base — competition day, season, or career
- Structured retrieval from Pinecone vector store using semantic search and metadata filters (by part, chapter, and content type)
- Questions outside the scope of the book trigger a graceful refusal rather than a speculative answer. This design choice prioritises reliability over breadth.
- LangSmith traces every call at the chain level.
- Inline citations in every response, formatted as [Dance To Your Maximum, Chapter X-X, pp. XX–XX], enabling athletes and coaches to verify and deepen their reading
- Faithfulness evaluation pipeline using an LLM-as-judge approach, targeting ≥90% faithfulness against a 15-question benchmark test set

## Value Proposition

- Trustworthy outputs: every claim links back to a specific page range in the source text, eliminating hallucination risk on in-scope questions
- Domain-aware retrieval: routing and metadata filtering ensure the model draws from the right section of the knowledge base, not just the nearest vector
- Measurable quality: the built-in evaluation pipeline gives objective faithfulness metrics before any deployment or knowledge-base update
- Low barrier to use: Streamlit UI requires no technical knowledge from end users

## Evaluation

15-question fixed test set with an LLM judge (`gpt-4.1-mini`, temp=0) plus manual spot-check. Target: ≥90% faithfulness (claims fully supported by retrieved context). PASS/FAIL is printed in-notebook.



