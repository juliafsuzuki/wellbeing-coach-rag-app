# DanceSport Wellbeing RAG Application

## Project Objective
> The RAG application, **DanceSport Wellbeing RAG Application** helps DanceSport athletes answer questions on showcase and competition preparation and performance readiness in a chatbot UI with grounded citations, safe fallback behavior, a target of 90% faithfulness, and under 8-second response time.

## Problem Statement

DanceSport athletes navigate a complex mix of physical, mental, emotional, psychological, and strategic challenges while preparing for showcase performances and competitions. While expert coaching and valuable educational resources exist, access to them is often limited and expensive. Many insights remain buried in books, articles, and other written materials, making discovery largely a matter of chance.

Athletes need timely access to trusted guidance when they need it most. To support effective training and performance, coaching knowledge should be readily accessible through a conversational, queryable experience.

## Solution Overview

The DanceSport Wellbeing Coach is a single-source Retrieval-Augmented Generation (RAG) application that enables DanceSport athletes to access trusted guidance on training, performance, and wellbeing by asking natural-language questions related to showcase and competition preparation.

The coach is designed to serve multiple roles, including a/an:
- Evidence-based advisor that provides trusted guidance specially for showcase and competition success.
- DanceSport coach that helps athletes train effectively, perform with confidence, and compete at their highest potential.
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

All responses are grounded exclusively in a PDF e-book, *Dance To Your Maximum* by Maximiliaan Winkelhuis, and include inline citations to the relevant chapter and page, enabling athletes to verify the source material and explore the concepts in greater depth.

### Corpus

*Dance To Your Maximum* (Maximiliaan Winkelhuis) is a structured workbook with:
- Part One: The Competition Day
- Part Two: The Season
- Part Three: The Dancer’s Career
- Numbered chapters such as 1-2 Stress and preparations, 1-4 The Nine Step Connection Model, 2-3 Planning your choreography, 2-8 Exercises during the season, and 3-1 Career planning
- Workbook sections including personal tests, questionnaires, evaluation forms, and appendices

### Scope and Constraints

- **Single corpus:** all knowledge is derived from one source — the e-book *Dance To Your Maximum*. No multi-document upload.
- **No multi-turn retrieval awareness:** each question is processed independently. Chat history is displayed in the UI but is not passed into the retrieval or generation steps.
- **No cross-session memory:** chat history clears on page reload.
- **No re-ranking or streaming:** retrieval returns top-k results by cosine similarity; responses are returned in full once generation completes.
- **Secrets via `.env`:** API keys are never hard-coded; the app runs as a single-process Streamlit deployment.

### Key capabilities

- **Conversational Q&A** via a Streamlit web interface, accessible to non-technical users
- **Intelligent query routing** (LangGraph) classifies each question on two dimensions — part scope (`competition_day`, `season`, `career`) and content type (`prose`, `test`, `form`, `appendix`) — and applies matching metadata filters before retrieval
- **Semantic retrieval** from Pinecone vector store with metadata filtering; falls back to unfiltered search when a filtered query returns no results
- **Image-based PDF support:** the source e-book is fully scanned (no text layer); PyMuPDF + Tesseract OCR pipeline extracts text with results cached to disk after the first run
- **Epistemic tagging:** every claim is labelled `[KNOWN]`, `[COMPUTED]`, `[INFERRED]`, `[COMMON]`, `[FRAME]`, or `[GUESS]`, with a confidence rating — making the reasoning transparent and auditable
- **Inline citations** in every response — `[Dance To Your Maximum, Chapter X-Y, Page Z–W]` — drawn from chunk metadata, never fabricated
- **Graceful refusal:** questions outside the scope of the retrieved context produce `"I don't have that in my knowledge base."` rather than a speculative answer
- **Faithfulness evaluation pipeline** using an LLM-as-judge (`gpt-4.1-mini`, temp=0) over a fixed 15-question benchmark, targeting ≥90% faithfulness
- **LangSmith observability** integrated for full chain tracing; currently disabled — activate by setting `LANGCHAIN_TRACING_V2=true` and supplying a `LANGCHAIN_API_KEY`

### Value Proposition

- **Trustworthy outputs:** every claim links back to a specific page range in the source text, minimising hallucination risk on in-scope questions. Out-of-scope questions are refused rather than answered speculatively.
- **Domain-aware retrieval:** two-dimensional routing (part scope + content type) and metadata filtering ensure the model draws from the right section of the knowledge base, not just the nearest vector.
- **Transparent reasoning:** epistemic tags and confidence ratings surface the basis for every claim, letting users judge answers rather than accept them at face value.
- **Measurable quality:** the built-in evaluation pipeline (Section 11) produces objective faithfulness metrics before any deployment or knowledge-base update.
- **Low barrier to use:** the Streamlit UI requires no technical knowledge; pre-built question categories make it immediately useful without any prior familiarity with the source material.
  
## Project Structure

```
dancesport-wellbeing-rag-app/
├── app.py                          # Streamlit chat UI
├── rag_chain.py                    # RAG graph, router, retriever, generator
├── 1_wellbeing_coach_rag_app_langchain.ipynb   # End-to-end pipeline + evaluation
├── generate_diagrams.py            # Diagram generation utility
├── requirements.txt                # Pinned direct dependencies
├── .gitignore
├── data/
│   └── ocr_cache.json              # Cached OCR output — avoids re-running Tesseract
│   # Note: the source PDF is not included (copyright). Add your own copy as:
│   # data/e-Book_dance-to-your-maximum.pdf
├── images/                         # Section process diagrams (01–11) + home page screenshot
├── images-2/                       # End-to-end workflow diagrams used in README
└── docs/
    ├── project_specification.md    # Functional & technical specification
    └── project_design.md           # Architecture & design decisions
```

### Documentation

- [Project Specification](docs/project_specification.md): User stories, UI categories, RAG graph, routing, metadata schema, evaluation pipeline, and constraints.
- [Project Design](docs/project_design.md): Architecture, chunking, retrieval, and evaluation decisions.

<br />

# How to access and test the RAG application

## 1. Prerequisites

Before running the app, make sure you have:

| Requirement | Notes |
|---|---|
| Python 3.9+ | [python.org](https://www.python.org/downloads/) |
| Tesseract OCR | [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki) · macOS: `brew install tesseract` · Linux: `sudo apt install tesseract-ocr` |
| OpenAI API key | [platform.openai.com](https://platform.openai.com/) |
| Pinecone API key | [pinecone.io](https://www.pinecone.io/) — free Starter plan is sufficient |
| Source PDF | *Dance To Your Maximum* by Maximiliaan Winkelhuis — place at `data/e-Book_dance-to-your-maximum.pdf` |

## 2. Accessing the App

**2.1. Clone the repository**

```bash
git clone https://github.com/juliafsuzuki/dancesport-wellbeing-rag-app.git
cd dancesport-wellbeing-rag-app
```

**2.2. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

**2.3. Configure environment variables**

Create a `.env` file in the project root with the following keys (never commit this file):

```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
LANGCHAIN_API_KEY=              # optional — leave blank to disable LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=wellbeing-coach-rag-app-langchain
```

**2.4. Build the Pinecone index**

Open and run the notebook end-to-end (Sections 1–10). This ingests the PDF via OCR, chunks the text, and uploads vectors to Pinecone. The OCR step takes several minutes on first run; results are cached to `data/ocr_cache.json` for all subsequent runs.

```bash
jupyter notebook 1_wellbeing_coach_rag_app_langchain.ipynb
```

> **Note:** Run the index-building step only once. Once the vectors are in Pinecone, you can skip straight to launching the app on future sessions.

**2.5. Launch the chatbot**

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

<br />

## 3. Testing the App

Once the app is running, here is how to explore it:

**Option A — Click a pre-built question**

The home page displays 6 question categories in a two-column grid. Click any question to send it immediately. Good questions to start with:

- *"How do I manage performance anxiety or stage fright?"*
- *"What is the best way to practice my showcase routine?"*
- *"How can visualization improve my showcase performance?"*

**Option B — Type your own question**

Use the chat input at the bottom of the page to ask anything related to competition preparation and performance readiness.

**What to expect in every response**

- Each factual claim is tagged: `[KNOWN]`, `[INFERRED]`, `[COMPUTED]`, `[COMMON]`, `[FRAME]`, or `[GUESS]`
- A confidence level (`HIGH` / `MED` / `LOW`) accompanies each claim
- An inline citation points to the exact chapter and page: `[Dance To Your Maximum, Chapter 1-2, Page 21–24]`
- If the question falls outside the book's scope, the response opens with `"I don't have that in my knowledge base."` — no speculation

**Option C — Run the evaluation pipeline**

Open the notebook and run **Section 11** to measure faithfulness across the fixed 15-question benchmark. A PASS/FAIL result is printed for each question, targeting ≥ 90% overall.

<br />

# Technical Overview

<img width="1672" height="941" alt="6-EndToEndWorkflow" src="https://github.com/user-attachments/assets/d0e37f8d-ef2b-4fdd-99c3-f166a12b1edb" />

<!-- START -->

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
source .venv/bin/activate       # macOS / Linux

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

1. PyMuPDF (`fitz`) renders each page to a grayscale image at 150 DPI.
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

### 6. Evaluation

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

<!-- END -->

<br />

# Chatbot UI

The home page surfaces 6 pre-built categories with clickable example questions, arranged in a two-column grid.

<img width="378" height="692" alt="Screenshot 2026-06-21 222442" src="https://github.com/user-attachments/assets/819e36f8-1ae6-4d46-ae61-886fcf705d5b" />

<br />

### Lists of questions for 6 categories:

**🎯 Performance Readiness**
- How do I know if I am ready to perform my showcase?
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

Every claim is tagged, confidence-rated, and cited.

**Epistemic tags** — applied to every claim:

| Tag | Meaning |
|---|---|
| `[KNOWN]` | Established training fact |
| `[COMPUTED]` | Calculated value |
| `[INFERRED]` | Logical deduction from context |
| `[COMMON]` | Standard domain knowledge |
| `[FRAME]` | Symbolic system — coherent within its frame, not a real-world claim |
| `[GUESS]` | No supporting basis |

**Confidence rating** — appended to each claim: `HIGH` (≥80%) · `MED` (50–80%) · `LOW` (20–50%) · `VERY LOW` (<20%) · `UNKNOWN`. `[FRAME]` and `[GUESS]` claims are capped at `LOW`.

**Citation** — inline after every claim: `[Dance To Your Maximum, Chapter 1-2, Page 21–24]`

**Refusal** — if the retrieved context does not contain a sufficient answer, the response opens with `"I don't have that in my knowledge base."` No speculation, no fabrication.

**Self-audit** — the model appends `[RULES I BROKE]: which, where, why` whenever it detects a rule violation in its own output.

<br />

## Evaluation

**Method:** automated LLM-as-judge (`gpt-4.1-mini`, temp=0) run inside the notebook (Section 11).

**Faithfulness** measures whether every factual claim in the generated answer is directly supported by the retrieved passages — it is not a measure of real-world accuracy. A faithful answer can only assert what the retrieved context contains.

| Setting | Value |
|---|---|
| Test set | 15 fixed questions, one per major topic area |
| Primary metric | Faithfulness — % of answers whose claims are fully supported by context |
| Target | ≥ 90% faithful answers |
| Secondary check | Manual spot-check on a representative subset |
| Output | Per-question PASS / FAIL with verdict printed in-notebook |



