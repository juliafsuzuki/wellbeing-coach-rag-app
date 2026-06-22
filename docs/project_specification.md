# Functional & Technical Specification
Functional & technical specification for the DanceSport Wellbeing Coach RAG app

## 1. Functional Specification

### 1.1 Target User

A DanceSport athlete (amateur or professional) preparing for a competition, showcase, or season. The user may have no AI/ML background; the interface must be self-explanatory.

### 1.2 User Stories

| ID | As a… | I want to… | So that… |
|---|---|---|---|
| US-01 | Athlete | Ask a free-text question in plain English | I can get relevant coaching advice immediately |
| US-02 | Athlete | Browse categorised example questions | I can discover what the system knows without knowing where to start |
| US-03 | Athlete | See which chapter and page the answer comes from | I can read the source material for deeper study |
| US-04 | Athlete | Receive an honest "I don't know" when the topic is outside the book | I am not misled by fabricated advice |
| US-05 | Athlete | Maintain a conversation history within a session | I can refer back to earlier answers without re-asking |

### 1.3 Question Categories (UI)

The home page (shown in the image below) surfaces 6 pre-built question categories with clickable example questions. The layout is a two-column grid.

<img width="378" height="692" alt="image" src="https://github.com/user-attachments/assets/8e576312-4a97-455e-a14a-4714b36bc1b7" />


- **Left column:** Performance Readiness, Practice & Preparation, Musicality & Timing
- **Right column:** Confidence & Stage Presence, Expression & Storytelling, Mindset & Mental Performance

#### 🎯 Performance Readiness

- How do I know if I am ready to perform my showcase?
- What should I focus on during the final weeks before my showcase?
- How can I reduce mistakes during my performance?
- What should I do if I forget part of my routine on the floor?

#### 📋 Practice & Preparation

- What is the best way to practice my showcase routine?
- How often should I practice between lessons?
- How do I remember my showcase routine more effectively?
- How can I make my practice sessions more productive?
- How do I transition from learning choreography to performing it?

#### 🎵 Musicality & Timing

- How can I improve my timing with the music?
- How do I stay on time when I get nervous?
- How can I better connect my movements to the music?
- How important is breathing for timing, movement, and performance quality?

#### ✨ Confidence & Stage Presence

- How can I look more confident on the dance floor?
- How do I manage performance anxiety or stage fright?
- How can I project confidence even when I feel nervous?
- How can I make a strong first impression when I enter the floor?

#### 💃 Expression & Storytelling

- How can I make my performance more expressive?
- How can I better tell the story of the dance?
- How do I connect emotionally with the music and my audience?
- What makes a showcase performance memorable?

#### 🧠 Mindset & Mental Performance

- Why is mental clarity important in DanceSport?
- How can I stay mentally focused during my performance?
- How do I recover mentally after making a mistake?
- How can visualization improve my showcase performance?

### 1.4 Answer Behaviour

- Every factual claim is tagged: `[KNOWN]`, `[INFERRED]`, `[COMPUTED]`, `[COMMON]`, `[FRAME]`, or `[GUESS]`
- Every claim carries an inline citation: `[Dance To Your Maximum, Chapter X-Y, Page Z–W]`
- When no relevant context is found, the first line of the answer is `"I don't have that in my knowledge base."` — no fabrication
- The persona is expert, direct, and anti-sycophantic: does not capitulate to pushback without new evidence
- A confidence level (HIGH / MED / LOW / VERY LOW / UNKNOWN) accompanies claims

### 1.5 Session Behaviour

- Chat history persists within a browser session (Streamlit `session_state`)
- History resets on page reload (no persistent cross-session memory)
- Clicking an example question injects it as if typed by the user

### 1.6 Evaluation Acceptance Criteria

- Faithfulness target: ≥ 90% of answers on the 15-question benchmark are fully supported by the retrieved context (LLM judge)
- Faithfulness is evaluated in-notebook; a PASS/FAIL result is printed

## 2. Technical Specification

### 2.1 Technology Stack

| Layer | Technology | Version / Model |
|---|---|---|
| Language | Python | 3.x (`.venv`) |
| LLM — routing & judge | OpenAI ChatOpenAI | `gpt-4.1-mini`, temp=0 |
| LLM — generation | OpenAI ChatOpenAI | `gpt-4.1-mini`, temp=0.1 |
| Embeddings | OpenAI | `text-embedding-3-small`, dim=1536 |
| Orchestration | LangChain + LangGraph | Current stable |
| Observability | LangSmith | Project: `wellbeing-coach-rag-app-langchain` |
| Vector store | Pinecone Serverless | Index: `wellbeing-coach-rag`, cosine, AWS `us-east-1` |
| PDF rendering | PyMuPDF (fitz) | — |
| OCR | Tesseract via pytesseract | English model |
| UI | Streamlit | — |
| Environment | python-dotenv / `.env` | Keys: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `LANGCHAIN_API_KEY` |

### 2.2 Repository Layout

```
dancesport-wellbeing-rag-app/
├── app.py                          # Streamlit chat UI
├── rag_chain.py                    # RAG graph, router, retriever, generator (importable)
├── 1_wellbeing_coach_rag_app_langchain.ipynb   # End-to-end pipeline + evaluation
├── generate_diagrams.py            # Diagram generation utility
├── requirements.txt                # Pinned direct dependencies
├── .gitignore
├── data/
│   └── ocr_cache.json              # Cached OCR output (run-once)
│   # Note: source PDF not included (copyright) — add as data/e-Book_dance-to-your-maximum.pdf
├── images/                         # Section process diagrams (01–11) + home page screenshot
├── images-2/                       # End-to-end workflow diagrams used in README
└── docs/
    ├── project_specification.md
    └── project_design.md
```

### 2.3 Data Pipeline

The ingestion pipeline runs once in the notebook and writes to Pinecone. The Streamlit app reads from Pinecone only.

```
PDF (316 image pages)
  │
  ▼  PyMuPDF → Tesseract OCR (150 DPI, grayscale)
  │  [cached to ocr_cache.json after first run]
  │
  ▼  Text Cleaning
  │  - merge hyphenated line-breaks
  │  - collapse whitespace
  │  - collapse 3+ blank lines → paragraph break
  │
  ▼  Structure Detection & Metadata Enrichment (per page)
  │  Regex-detected fields:
  │    part / part_scope   (Part One → competition_day, etc.)
  │    chapter / chapter_title  (e.g. "1-2", "Season Planning")
  │    section_type        (prose | test | form | appendix | intro | toc)
  │    page_start, page_end
  │    source = "dance_to_your_maximum"
  │
  ▼  Hierarchical Chunking (RecursiveCharacterTextSplitter)
  │    Prose:        chunk_size=1200, overlap=200
  │    Test/Form:    chunk_size=2200, overlap=150
  │    min_chunk_size=100 (discard below)
  │    chunk_id = MD5 hash of content (12 chars)
  │    parent_id = "page_{page_start}"
  │
  ▼  OpenAI text-embedding-3-small  (1536-dim vectors)
  │
  ▼  Pinecone upsert (batch=100)
     Index: wellbeing-coach-rag
```

### 2.4 Chunk Metadata Schema (Pinecone)

Each vector carries this metadata payload:

| Field | Type | Example |
|---|---|---|
| `source` | str | `"dance_to_your_maximum"` |
| `part` | str | `"Part One"` |
| `part_scope` | str | `"competition_day"` |
| `chapter` | str | `"1-2"` |
| `chapter_title` | str | `"Competition Morning Routine"` |
| `section_type` | str | `"prose"` |
| `page_start` | int | 21 |
| `page_end` | int | 24 |
| `chunk_id` | str | `"a3f9bc12de7f"` |
| `parent_id` | str | `"page_21"` |

### 2.5 RAG Graph (LangGraph)

The runtime pipeline is a deterministic 3-node directed graph with no conditional branches:

```
[route] ──► [retrieve] ──► [generate] ──► END
```

**State (`RAGState` TypedDict):**

| Field | Type | Set by |
|---|---|---|
| `question` | str | caller |
| `chat_history` | List[BaseMessage] | caller |
| `route` | dict | `route` node |
| `documents` | List[Document] | `retrieve` node |
| `answer` | str | `generate` node |

**Node: `route`**
- Calls `gpt-4.1-mini` with a structured prompt
- Returns `{"part_scope": "...", "section_type": "..."}` (JSON)
- Fallback on parse error: `{"part_scope": "general", "section_type": "general"}`

**Node: `retrieve`**
- Builds a Pinecone filter from the route
  - `part_scope != "general"` → `{"part_scope": {"$eq": value}}`
  - `section_type` not in `{"prose", "general"}` → `{"section_type": {"$in": [value]}}`
- Primary search: `k=4` with filter; fallback to `k=6` unfiltered if filter returns 0 results
- Returns `documents` list

**Node: `generate`**
- Formats context with inline source attribution per chunk
- Calls `gpt-4.1-mini` (temp=0.1) with system prompt + context + question
- Returns answer string
- If `documents` is empty → returns refusal string directly (no LLM call)

### 2.6 Query Routing Logic

| Question intent | `part_scope` | `section_type` |
|---|---|---|
| Competition day stress, morning routine, warm-up | `competition_day` | `prose` |
| Season planning, choreography, exercises | `season` | `prose` |
| Long-term goals, career arc, identity | `career` | `prose` |
| Questionnaire, self-assessment, personal test | (any) | `test` |
| Evaluation form, worksheet | (any) | `form` |
| Appendix content | (any) | `appendix` |
| Unclear or cross-part | `general` | `general` |

When `general` is returned, no Pinecone filter is applied; all 6 top-k results come from the full index.

### 2.7 Answer Format & Epistemic Tags

The system prompt imposes a strict tagging and citation protocol:

```
[KNOWN]    — established training fact
[INFERRED] — logical deduction from context
[COMPUTED] — numeric calculation
[COMMON]   — standard field knowledge
[FRAME]    — symbolic system (coherent but not empirical)
[GUESS]    — no basis in context

Confidence: HIGH ≥80% · MED 50–80% · LOW 20–50% · VERY LOW <20% · UNKNOWN

Citation:  [Dance To Your Maximum, Chapter X-Y, pp. Z–W]
```

### 2.8 Streamlit UI Architecture

`app.py` imports the `ask()` function from `rag_chain.py`. The RAG graph is initialised once via a module-level singleton (`_rag_app`) and reused across all requests.

**UI layout (see [`images/`](../images/)):**
- Page header: "DanceSport Wellbeing Coach" with logo and link icon
- Subtitle: "Grounded in *Dance To Your Maximum* — Maximiliaan Winkelhuis"
- Description paragraph: "Ask me anything about showcase or competition preparation. I'll answer using the coaching materials in my knowledge base and cite the exact passage and page number for every response."
- Horizontal divider
- Prompt heading: "Ask me anything — or pick a question below to get started:"
- 2-column grid of 6 category headers (each with an emoji icon) and clickable question buttons:
  - Left column: Performance Readiness (4), Practice & Preparation (5), Musicality & Timing (4)
  - Right column: Confidence & Stage Presence (4), Expression & Storytelling (4), Mindset & Mental Performance (4)
- Scrollable chat history (Streamlit `st.chat_message`)
- Persistent `st.chat_input` at the bottom ("Ask your question…")

**Pending question pattern:** clicking a button sets `st.session_state.pending_question`; the main render loop reads it before reading `st.chat_input`, then calls `st.rerun()` to re-render with the answer appended.

### 2.9 Observability

All LLM calls are traced to LangSmith automatically when `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` are set. The project name is `wellbeing-coach-rag-app-langchain`. Each full invocation (route + retrieve + generate) appears as a single parent trace with three child spans.

### 2.10 Evaluation Pipeline

Located in notebook Section 11. Runs in-notebook against the live Pinecone index.

- Test set: 15 fixed questions covering all 3 parts and both prose and form section types
- Judge model: `gpt-4.1-mini`, temp=0
- Verdict format: `FAITHFUL: <reason>` or `NOT_FAITHFUL: <reason>`
- Pass threshold: ≥ 90% faithful (≥ 14 / 15)
- Failures are printed with question, truncated answer, and judge reasoning

## 3. Constraints & Non-Requirements

| Constraint | Detail |
|---|---|
| Single corpus | The system is hard-coded to *Dance To Your Maximum*; no multi-document or upload capability |
| No cross-session memory | Chat history lives in `st.session_state` only; reloading the page clears it |
| No re-ranking | Retrieval is pure dense cosine similarity; no BM25 hybrid or cross-encoder re-ranking |
| No streaming | Answers are returned as a complete string; no token-by-token streaming to the UI |
| OCR quality dependency | Answer quality is bounded by Tesseract OCR accuracy on scanned pages; tables and diagrams may not OCR cleanly |
| API key management | Keys are loaded from a local `.env` file; no secrets manager or rotation mechanism |
| Single-user deployment | Streamlit runs as a single-process local server; no authentication or multi-tenancy |
