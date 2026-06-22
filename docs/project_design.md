# Design Decisions
Full architecture, chunking, retrieval, and evaluation decisions for DanceSport Wellbeing Coach RAG app (LangChain + Pinecone + OpenAI + Streamlit)

## Stack
- LangChain + LangGraph + LangSmith
- OpenAI: gpt-4.1-mini (ChatOpenAI), text-embedding-3-small (OpenAIEmbeddings, dim=1536)
- Pinecone: cosine similarity, serverless
- UI: Streamlit (app.py)
- Notebook: 1_wellbeing_coach__rag_app_langchain.ipynb
- LangSmith project name: wellbeing-coach-rag-app-langchain

## Corpus
Single PDF: "Dance To Your Maximum" by Maximiliaan Winkelhuis
Path: C:\Users\julia\wellbeing-coach-rag-app\data\e-Book_dance-to-your-maximum.pdf
Structure: workbook with Part One (Competition Day), Part Two (Season), Part Three (Career)
Numbered chapters (e.g. 1-2, 1-4, 2-3, 2-8, 3-1), tests, forms, questionnaires, appendices

## Ingestion
- Load page by page with PyPDFLoader
- Preserve page numbers for citations
- Clean OCR conservatively: normalize whitespace, merge hyphenated line breaks, remove repeated headers/footers
- Detect structure via regex: Part headings, Chapter headings, section_type

## Chunking (hierarchical)
- Prose chapters: chunk_size=1200, chunk_overlap=200
- Tests/forms/appendices: chunk_size=2200, chunk_overlap=150
- min_chunk_size=350 (discard smaller)
- RecursiveCharacterTextSplitter per section type

## Metadata per chunk
- source = dance_to_your_maximum
- part (e.g. "Part One")
- part_scope = competition_day | season | career
- chapter (e.g. "1-2")
- chapter_title
- section_type = prose | test | form | appendix | intro | toc
- page_start, page_end
- chunk_id, parent_id

## Retrieval
- V1: pure dense retrieval
- top_k=6 default, filtered_top_k=4 when route detected
- Metadata filters: part_scope and/or section_type
- Refusal path when retrieval is weak

## Query routing (LangGraph node)
- competition-day questions → part_scope=competition_day
- season/exercise questions → part_scope=season
- career/goals questions → part_scope=career
- questionnaire/form/test → section_type in {test, form, appendix}

## Citations
Format: [Dance To Your Maximum, Chapter 1-2, pp. 21–24]
Inline in answer body after each claim.

## Evaluation
- 15-question fixed test set
- LLM judge (automated)
- Manual spot-check subset
- Target: ≥90% faithfulness (claims fully supported by retrieved context)
- In-notebook eval pipeline

## Environment
- .env file: user fills manually with OPENAI_API_KEY, PINECONE_API_KEY, LANGCHAIN_API_KEY
- Virtual environment: .venv
