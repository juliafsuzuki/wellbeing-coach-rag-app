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

All responses are grounded exclusively in Dance To Your Maximum by Maximiliaan Winkelhuis and include inline citations to the relevant chapter and page, enabling athletes to verify the source material and explore the concepts in greater depth.

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

## Technical Architecture

<img width="1672" height="941" alt="6-EndToEndWorkflow" src="https://github.com/user-attachments/assets/d0e37f8d-ef2b-4fdd-99c3-f166a12b1edb" />
<br />

### Getting started:
1. Create a virtual environment: `python -m venv .venv`
2. Activate it and install dependencies.
3. Populate `.env` with `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `LANGCHAIN_API_KEY`.
4. Run the notebook to ingest the PDF and build the Pinecone index.
5. Launch the UI: `streamlit run app.py`

<br />

### AI Stack:
- **Orchestration:** LangChain + LangGraph + LangSmith
- **LLM:** OpenAI `gpt-4.1-mini` (routing/judge temp=0, generation temp=0.1)
- **Embeddings:** OpenAI `text-embedding-3-small` (dim=1536)
- **Vector store:** Pinecone Serverless — index `wellbeing-coach-rag`, cosine, AWS `us-east-1`
- **PDF + OCR:** PyMuPDF (fitz) + Tesseract (pytesseract)
- **UI:** Streamlit (`app.py`)
- **Notebook:** `1_wellbeing_coach__rag_app_langchain.ipynb`
- **LangSmith project:** `wellbeing-coach-rag-app-langchain`

<img width="275" height="134" alt="image" src="https://github.com/user-attachments/assets/ad6e478c-c8af-439f-b833-44c4a4a02228" />

The ingestion pipeline applies hierarchical chunking tuned to the workbook's structure: tighter chunks for prose chapters (1,200 chars) and larger windows for tests, forms, and appendices (2,200 chars), preserving semantic coherence. Each chunk carries rich metadata — part scope, chapter, section type, and page range — enabling targeted retrieval and accurate citation generation.

<br />

## Home page / GUI

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

<br />

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
- Inline citations in every response, formatted as [Dance To Your Maximum, Chapter X-X, pp. XX–XX], enabling athletes and coaches to verify and deepen their reading
- Faithfulness evaluation pipeline using an LLM-as-judge approach, targeting ≥90% faithfulness against a 15-question benchmark test set

## Value Proposition

- Trustworthy outputs: every claim links back to a specific page range in the source text, eliminating hallucination risk on in-scope questions
- Domain-aware retrieval: routing and metadata filtering ensure the model draws from the right section of the knowledge base, not just the nearest vector
- Measurable quality: the built-in evaluation pipeline gives objective faithfulness metrics before any deployment or knowledge-base update
- Low barrier to use: Streamlit UI requires no technical knowledge from end users

## Evaluation

15-question fixed test set with an LLM judge (`gpt-4.1-mini`, temp=0) plus manual spot-check. Target: ≥90% faithfulness (claims fully supported by retrieved context). PASS/FAIL is printed in-notebook.



