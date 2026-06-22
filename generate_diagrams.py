"""Generate section diagrams for the Wellbeing Coach RAG notebook."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('images', exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    bg='#f5f3ff', purple='#7c3aed', lp='#c4b5fd', lav='#ede9fe',
    dark='#1e1b4b', gray='#6b7280', body='#374151', white='#ffffff',
    cbg='#1e1b4b', cfg='#c7d2fe', green='#34d399', red='#fca5a5',
)

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_card(ax, x, y, w, h, num, title, lines, tags=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015',
                 lw=2, ec=C['lp'], fc=C['white'], zorder=2))
    ax.add_patch(plt.Circle((x+0.048, y+h-0.077), 0.033, color=C['purple'], zorder=3))
    ax.text(x+0.048, y+h-0.077, str(num), ha='center', va='center',
            color='white', fontsize=9, fontweight='bold', zorder=4)
    ax.text(x+0.097, y+h-0.082, title, ha='left', va='center',
            color=C['dark'], fontsize=9, fontweight='bold', zorder=3)
    ax.add_patch(FancyBboxPatch((x+0.018, y+0.10), w-0.036, h-0.20,
                 boxstyle='round,pad=0.008', lw=0, fc=C['lav'], zorder=2))
    ly = y + h - 0.195
    for ln in lines:
        ax.text(x+0.030, ly, ln, ha='left', va='top',
                color=C['body'], fontsize=7.8, zorder=3)
        ly -= 0.048
    if tags:
        tx = x + 0.025
        for t in tags:
            tw = max(len(t) * 0.009 + 0.022, 0.065)
            ax.add_patch(FancyBboxPatch((tx, y+0.015), tw, 0.052,
                         boxstyle='round,pad=0.004', lw=0, fc=C['lav'], zorder=3))
            ax.text(tx+tw/2, y+0.041, t, ha='center', va='center',
                    color=C['purple'], fontsize=7, zorder=4)
            tx += tw + 0.012


def draw_arrow(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=C['purple'], lw=2.5), zorder=2)


def draw_code(ax, x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                 lw=0, fc=C['cbg'], zorder=2))
    cy = y + h - 0.044
    for ln in lines:
        ax.text(x+0.018, cy, ln, ha='left', va='top', color=C['cfg'],
                fontsize=7.8, fontfamily='monospace', zorder=3)
        cy -= 0.041


def draw_tip(ax, x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                 lw=2, ec=C['lp'], fc=C['white'], zorder=2))
    ax.text(x+0.022, y+h-0.052, 'Keep in mind:', ha='left', va='top',
            color=C['dark'], fontsize=9, fontweight='bold', zorder=3)
    ty = y + h - 0.108
    for ln in lines:
        ax.text(x+0.022, ty, ln, ha='left', va='top',
                color=C['body'], fontsize=7.8, zorder=3)
        ty -= 0.044


def make(title, subtitle, steps, code, tip, fname):
    fig = plt.figure(figsize=(11.5, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    fig.patch.set_facecolor(C['bg']); ax.set_facecolor(C['bg'])

    ax.text(0.025, 0.965, title, ha='left', va='top',
            fontsize=16, fontweight='bold', color=C['dark'])
    ax.text(0.025, 0.906, subtitle, ha='left', va='top',
            fontsize=9.5, color=C['gray'])
    ax.plot([0.025, 0.975], [0.878, 0.878], color=C['lp'], lw=1)

    # 3-card fixed layout
    xs = [0.025, 0.360, 0.695]
    cw, ch, cy = 0.300, 0.415, 0.435
    for i, (s, px) in enumerate(zip(steps, xs)):
        draw_card(ax, px, cy, cw, ch, i+1, s['title'], s['lines'], s.get('tags'))
        if i < 2:
            draw_arrow(ax, px+cw+0.008, xs[i+1]-0.008, cy+ch/2)

    draw_code(ax, 0.025, 0.020, 0.625, 0.385, code)
    draw_tip(ax,  0.665, 0.020, 0.310, 0.385, tip)

    fig.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=C['bg'], pad_inches=0.04)
    plt.close(fig)
    print(f'  Saved: {fname}')


# ── Diagram 1: Setup & Dependencies ──────────────────────────────────────────
make(
    title='1. Setup & Dependencies',
    subtitle='Create an isolated Python environment and install all required packages',
    steps=[
        dict(title='Create Virtual Environment', lines=[
            'python -m venv .venv',
            '',
            'Python 3.11.1',
            'Packages isolated from',
            'the rest of your system',
        ], tags=['.venv', 'Python 3.11']),
        dict(title='Install Packages', lines=[
            'langchain   langchain-openai',
            'langchain-pinecone  langgraph',
            'langsmith   pinecone',
            'pymupdf     pytesseract',
            'streamlit   python-dotenv',
        ], tags=['pip install']),
        dict(title='Register Jupyter Kernel', lines=[
            'ipykernel install',
            '  --name wellbeing-coach-rag',
            '',
            'Select in VS Code:',
            '"Python (.venv)"',
        ], tags=['ipykernel', 'Jupyter']),
    ],
    code=[
        'python -m venv .venv',
        '.venv\\Scripts\\pip install langchain langchain-openai \\',
        '    langchain-pinecone langgraph langsmith pinecone \\',
        '    pymupdf pytesseract streamlit python-dotenv ipykernel',
        '',
        '.venv\\Scripts\\python -m ipykernel install --user \\',
        '    --name=wellbeing-coach-rag \\',
        '    --display-name="Python (.venv)"',
    ],
    tip=[
        'Always select the',
        '"Python (.venv)" kernel',
        'before running cells.',
        '',
        'This ensures every import',
        'resolves to packages',
        'installed in this project,',
        'not your global Python.',
    ],
    fname='images/01_setup_dependencies.png',
)

# ── Diagram 2: Environment & LangSmith ───────────────────────────────────────
make(
    title='2. Environment & LangSmith',
    subtitle='Load API keys from .env and configure observability tracing',
    steps=[
        dict(title='The .env File', lines=[
            'OPENAI_API_KEY=sk-...',
            'PINECONE_API_KEY=pcsk_...',
            'LANGCHAIN_API_KEY=ls__...',
            'LANGCHAIN_TRACING_V2=true',
            'LANGCHAIN_PROJECT=...',
        ], tags=['.env', 'secrets']),
        dict(title='Load with python-dotenv', lines=[
            'load_dotenv()',
            '',
            'Reads .env file and',
            'injects every KEY=VALUE',
            'into os.environ',
        ], tags=['load_dotenv()']),
        dict(title='Verify & Enable Tracing', lines=[
            'OPENAI key    : True',
            'PINECONE key  : True',
            'LangSmith project set',
            '',
            'Every chain call is now',
            'logged to LangSmith',
        ], tags=['LangSmith', 'tracing']),
    ],
    code=[
        'from dotenv import load_dotenv',
        'import os',
        '',
        'load_dotenv()   # reads .env into environment',
        '',
        'OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")',
        'PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")',
        '',
        '# LangSmith activates automatically via env vars',
        '# LANGCHAIN_TRACING_V2=true  LANGCHAIN_PROJECT=...',
    ],
    tip=[
        'Never commit .env to git.',
        'Add it to .gitignore.',
        '',
        'LangSmith traces every',
        'chain call so you can',
        'debug retrieval and',
        'generation step-by-step',
        'in the web dashboard.',
    ],
    fname='images/02_environment_langsmith.png',
)

# ── Diagram 3: PDF Ingestion (OCR) ────────────────────────────────────────────
make(
    title='3. PDF Ingestion — OCR Pipeline',
    subtitle='Turn a scanned, image-based PDF into LangChain Documents using PyMuPDF + Tesseract',
    steps=[
        dict(title='Load PDF with PyMuPDF', lines=[
            'fitz.open(pdf_path)',
            '',
            '316 pages loaded',
            'Each page is a scanned',
            'image — no text layer',
        ], tags=['PyMuPDF', 'fitz', 'image-based']),
        dict(title='OCR Each Page', lines=[
            'Render page to grayscale',
            'image at 150 DPI',
            '',
            'Tesseract reads the image',
            'and extracts text',
            'Results cached to disk',
        ], tags=['Tesseract', '150 DPI', 'cache']),
        dict(title='Wrap as LangChain Document', lines=[
            'Document(',
            '  page_content = ocr_text,',
            '  metadata = {',
            '    "source": "dance_to...",',
            '    "page_num": i + 1 })',
        ], tags=['Document', '316 objects']),
    ],
    code=[
        'for i in range(len(doc)):',
        '    pix  = doc[i].get_pixmap(matrix=mat,',
        '                             colorspace=fitz.csGRAY)',
        '    img  = Image.frombytes("L",',
        '               (pix.width, pix.height), pix.samples)',
        '    text = pytesseract.image_to_string(img, lang="eng")',
        '    raw_pages.append(Document(',
        '        page_content=text,',
        '        metadata={"source": "dance_to_your_maximum",',
        '                  "page_num": i + 1}))',
    ],
    tip=[
        'OCR runs once then saves',
        'to ocr_cache.json.',
        '',
        'Every subsequent run',
        'loads from cache in',
        'seconds — no GPU,',
        'no API cost,',
        'no waiting.',
    ],
    fname='images/03_pdf_ingestion_ocr.png',
)

# ── Diagram 4: Text Cleaning ──────────────────────────────────────────────────
make(
    title='4. Text Cleaning',
    subtitle='Fix common OCR artefacts conservatively — preserve all chapter numbers and headings',
    steps=[
        dict(title='Fix Hyphenated Line Breaks', lines=[
            'OCR often breaks words',
            'across lines with a dash:',
            '',
            '"danc-',
            ' ing" -> "dancing"',
        ], tags=['regex', 'hyphen fix']),
        dict(title='Normalise Whitespace', lines=[
            'Multiple spaces and tabs',
            'become a single space:',
            '',
            '"hello   world"',
            '-> "hello world"',
        ], tags=['whitespace', 'tabs']),
        dict(title='Collapse Blank Lines', lines=[
            '3 or more consecutive',
            'blank lines collapse to',
            'a single paragraph break:',
            '',
            'Preserves structure',
        ], tags=['paragraph', 'newlines']),
    ],
    code=[
        'def clean_text(text: str) -> str:',
        '    # Merge hyphenated line breaks',
        '    text = re.sub(r"(\\w)-\\n(\\w)", r"\\1\\2", text)',
        '    # Collapse spaces and tabs',
        '    text = re.sub(r"[ \\t]+", " ", text)',
        '    # Max one blank line between paragraphs',
        '    text = re.sub(r"\\n{3,}", "\\n\\n", text)',
        '    return text.strip()',
        '',
        'pages_cleaned = [clean_text(p.page_content) for p in raw_pages]',
    ],
    tip=[
        'Cleaning is conservative',
        'on purpose.',
        '',
        'Chapter numbers like',
        '"1-2" and "2-8" must',
        'survive intact — they',
        'are used later for',
        'metadata detection.',
    ],
    fname='images/04_text_cleaning.png',
)

# ── Diagram 5: Structure Detection & Metadata ─────────────────────────────────
make(
    title='5. Structure Detection & Metadata',
    subtitle='Scan each page for headings and section markers, then attach rich metadata to every page',
    steps=[
        dict(title='Detect Part Headings', lines=[
            '"Part One"  -> competition_day',
            '"Part Two"  -> season',
            '"Part Three"-> career',
            '',
            'State carried forward',
            'to all following pages',
        ], tags=['regex', 'part_scope']),
        dict(title='Detect Chapter Numbers', lines=[
            'Pattern: "1-2 Stress..."',
            '         "2-8 Exercises..."',
            '',
            'Extracts chapter number',
            'and chapter title for',
            'citation metadata',
        ], tags=['chapter', 'chapter_title']),
        dict(title='Classify Section Type', lines=[
            'prose   — main text',
            'test    — questionnaire',
            'form    — eval form',
            'appendix— appendix',
            'intro   — introduction',
        ], tags=['section_type', 'metadata']),
    ],
    code=[
        'for page in pages_cleaned:',
        '    part, scope  = detect_part(page.page_content)',
        '    chapter, ttl = detect_chapter(page.page_content)',
        '    stype        = detect_section_type(page.page_content)',
        '    page.metadata.update({',
        '        "part": part, "part_scope": scope,',
        '        "chapter": chapter, "chapter_title": ttl,',
        '        "section_type": stype,',
        '        "page_start": page_num, "page_end": page_num,',
        '    })',
    ],
    tip=[
        'Metadata travels with',
        'every chunk into',
        'Pinecone.',
        '',
        'At retrieval time it',
        'powers smart filters:',
        '"only search Part Two"',
        'or "only fetch forms".',
    ],
    fname='images/05_structure_detection.png',
)

# ── Diagram 6: Hierarchical Chunking ─────────────────────────────────────────
make(
    title='6. Hierarchical Chunking',
    subtitle='Split pages into retrieval-sized pieces using different rules for prose vs workbook forms',
    steps=[
        dict(title='Choose Splitter by Type', lines=[
            'prose / intro / toc:',
            '  chunk_size    = 1200',
            '  chunk_overlap = 200',
            '',
            'test / form / appendix:',
            '  chunk_size    = 2200',
            '  chunk_overlap = 150',
        ], tags=['RecursiveCharacterTextSplitter']),
        dict(title='Split Pages into Chunks', lines=[
            'Splits on paragraph',
            'breaks first, then',
            'sentences, then words',
            '',
            'Overlap keeps context',
            'across chunk boundaries',
        ], tags=['overlap', 'separators']),
        dict(title='Filter & Tag Chunks', lines=[
            'Discard chunks < 100 chars',
            '(blank pages, headers)',
            '',
            'Add chunk_id  (MD5 hash)',
            'Add parent_id (page ref)',
        ], tags=['chunk_id', 'parent_id', 'min 100 chars']),
    ],
    code=[
        'for page in enriched_pages:',
        '    stype    = page.metadata["section_type"]',
        '    splitter = FORM_SPLITTER if stype in FORM_TYPES \\',
        '               else PROSE_SPLITTER',
        '    for chunk in splitter.split_documents([page]):',
        '        if len(chunk.page_content) < MIN_CHUNK_SIZE:',
        '            continue   # discard very small fragments',
        '        chunk.metadata["chunk_id"]  = md5(chunk.page_content)',
        '        chunk.metadata["parent_id"] = f"page_{page_num}"',
        '        all_chunks.append(chunk)',
    ],
    tip=[
        'Larger chunks for forms',
        'keep questionnaire',
        'questions and answer',
        'spaces together.',
        '',
        'Smaller prose chunks',
        'improve precision when',
        'retrieving specific',
        'coaching advice.',
    ],
    fname='images/06_hierarchical_chunking.png',
)

# ── Diagram 7: Pinecone Indexing ──────────────────────────────────────────────
make(
    title='7. Pinecone Indexing',
    subtitle='Embed every chunk into a 1536-dimension vector and store in a Pinecone vector database',
    steps=[
        dict(title='Create Pinecone Index', lines=[
            'Index: wellbeing-coach-rag',
            'Dimensions: 1536',
            'Metric: cosine similarity',
            '',
            'Created once;',
            'reused on every run',
        ], tags=['Pinecone', 'cosine', 'serverless']),
        dict(title='Embed with OpenAI', lines=[
            'text-embedding-3-small',
            '1536-dim dense vectors',
            '',
            'Each chunk text ->',
            'a point in 1536-D space',
            'Similar text = near points',
        ], tags=['OpenAIEmbeddings', '1536 dims']),
        dict(title='Upsert in Batches', lines=[
            'Upload 100 chunks/batch',
            'with all metadata fields',
            '',
            'Run ONCE then comment',
            'out to avoid re-indexing',
            'on every notebook run',
        ], tags=['batch=100', 'run once']),
    ],
    code=[
        'embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")',
        'vectorstore = PineconeVectorStore(',
        '    index_name=INDEX_NAME, embedding=embeddings)',
        '',
        'def upsert_chunks(chunks, vectorstore, batch_size=100):',
        '    for i in range(0, len(chunks), batch_size):',
        '        batch = chunks[i : i + batch_size]',
        '        vectorstore.add_documents(batch)',
        '        print(f"Upserted {i+batch_size}/{len(chunks)}")',
    ],
    tip=[
        'Comment out upsert_chunks()',
        'after the first run.',
        '',
        'Re-running it will',
        'create duplicate vectors',
        'and inflate your',
        'Pinecone usage.',
        '',
        'The index persists',
        'in Pinecone cloud.',
    ],
    fname='images/07_pinecone_indexing.png',
)

# ── Diagram 8: Query Router ───────────────────────────────────────────────────
make(
    title='8. Query Router',
    subtitle='Classify each user question so the retriever knows which part of the book to search first',
    steps=[
        dict(title='Receive User Question', lines=[
            'Natural language question',
            'from the Streamlit UI',
            'or notebook test cell',
            '',
            '"How do I manage stress',
            ' on competition morning?"',
        ], tags=['user input']),
        dict(title='LLM Classifier', lines=[
            'gpt-4.1-mini reads the',
            'question and returns JSON:',
            '',
            '{"part_scope":',
            '  "competition_day",',
            ' "section_type": "prose"}',
        ], tags=['gpt-4.1-mini', 'JSON output']),
        dict(title='Route to Metadata Filter', lines=[
            'part_scope -> filter by',
            '  competition_day / season',
            '  / career / general',
            '',
            'section_type -> filter by',
            '  prose/test/form/appendix',
        ], tags=['metadata filter', 'top_k=4']),
    ],
    code=[
        'ROUTER_PROMPT = ChatPromptTemplate.from_messages([',
        '    ("system", """Classify into part_scope and',
        '       section_type. Return JSON only.',
        '       Example: {{"part_scope": "season",',
        '                  "section_type": "prose"}}"""),',
        '    ("human", "Query: {question}"),',
        '])',
        '',
        'def route_query(question):',
        '    result = (ROUTER_PROMPT | llm | StrOutputParser())',
        '    return json.loads(result.invoke({"question": question}))',
    ],
    tip=[
        'Routing improves precision.',
        '',
        'A question about',
        '"competition morning"',
        'should only search',
        'Part One — not the',
        'career planning',
        'chapters in Part Three.',
    ],
    fname='images/08_query_router.png',
)

# ── Diagram 9: RAG Chain — LangGraph ─────────────────────────────────────────
make(
    title='9. RAG Chain — LangGraph',
    subtitle='A stateful three-node graph that routes, retrieves, and generates a grounded answer',
    steps=[
        dict(title='Node 1: route', lines=[
            'Calls route_query()',
            'Returns part_scope and',
            'section_type',
            '',
            'Builds Pinecone metadata',
            'filter from the route',
        ], tags=['LangGraph node', 'filter']),
        dict(title='Node 2: retrieve', lines=[
            'Similarity search with',
            'metadata filter applied',
            '',
            'top_k = 4 (filtered)',
            'top_k = 6 (unfiltered)',
            'Falls back if 0 results',
        ], tags=['Pinecone', 'similarity search']),
        dict(title='Node 3: generate', lines=[
            'Formats retrieved chunks',
            'with source citations',
            '',
            'System prompt + context',
            '+ question -> gpt-4.1-mini',
            '-> tagged, cited answer',
        ], tags=['gpt-4.1-mini', 'citations']),
    ],
    code=[
        'class RAGState(TypedDict):',
        '    question: str',
        '    route: dict',
        '    documents: List[Document]',
        '    answer: str',
        '',
        'graph = StateGraph(RAGState)',
        'graph.add_node("route",    route_node)',
        'graph.add_node("retrieve", retrieve_node)',
        'graph.add_node("generate", generate_node)',
        'graph.set_entry_point("route")',
        'graph.add_edge("route","retrieve")',
        'graph.add_edge("retrieve","generate")',
    ],
    tip=[
        'LangGraph manages state',
        'as data flows through',
        'the three nodes.',
        '',
        'Each node receives',
        'the full state dict',
        'and returns only the',
        'fields it updates.',
    ],
    fname='images/09_rag_chain_langgraph.png',
)

# ── Diagram 10: Interactive Test ──────────────────────────────────────────────
make(
    title='10. Interactive Test',
    subtitle='Send live questions through the full RAG pipeline and inspect grounded answers',
    steps=[
        dict(title='Submit a Question', lines=[
            'ask("How do I manage',
            '    stress on competition',
            '    morning?")',
            '',
            'Invokes the full LangGraph',
            'graph end-to-end',
        ], tags=['ask()', 'live query']),
        dict(title='Chain Executes', lines=[
            'route   -> competition_day',
            'retrieve-> top 4 chunks',
            '          from Part One',
            'generate-> gpt-4.1-mini',
            '          writes answer',
        ], tags=['LangSmith trace']),
        dict(title='Grounded Answer Returned', lines=[
            'Every claim tagged:',
            '[KNOWN] [INFERRED]',
            '[COMMON] etc.',
            '',
            'Inline citation after',
            'each claim:',
            '[Dance To Your Maximum,',
            ' Chapter 1-2, Page 21-24]',
        ], tags=['citations', 'tags']),
    ],
    code=[
        'def ask(question, chat_history=None):',
        '    result = rag_app.invoke({',
        '        "question":     question,',
        '        "chat_history": chat_history or [],',
        '        "route":        {},',
        '        "documents":    [],',
        '        "answer":       "",',
        '    })',
        '    return result["answer"]',
        '',
        'print(ask("How do I manage stress on competition morning?"))',
    ],
    tip=[
        'Run 2-3 test questions',
        'that span all three',
        'parts of the book:',
        '',
        'competition day,',
        'season planning,',
        'and career.',
        '',
        'Check citations match',
        'the right chapters.',
    ],
    fname='images/10_interactive_test.png',
)

# ── Diagram 11: Faithfulness Evaluation ──────────────────────────────────────
make(
    title='11. Faithfulness Evaluation',
    subtitle='Measure how well the RAG chain stays grounded in the book using an automated LLM judge',
    steps=[
        dict(title='Fixed 15-Question Test Set', lines=[
            'Covers all 3 parts:',
            '  competition day',
            '  season planning',
            '  career & identity',
            '',
            'Same set on every run',
            'for reproducibility',
        ], tags=['15 questions', 'fixed set']),
        dict(title='Run RAG Chain + Collect', lines=[
            'For each question:',
            '  invoke rag_app()',
            '  capture answer',
            '  capture retrieved docs',
            '  format context',
        ], tags=['rag_app.invoke()', 'context']),
        dict(title='LLM Judge Scores Each', lines=[
            'gpt-4.1-mini reads:',
            '  context + question',
            '  + answer',
            '',
            'Returns FAITHFUL or',
            'NOT_FAITHFUL + reason',
            'Target: >= 90% FAITHFUL',
        ], tags=['LLM judge', 'target 90%']),
    ],
    code=[
        'JUDGE_PROMPT = ChatPromptTemplate.from_messages([',
        '    ("system", "Is every claim in the answer',
        '       supported by the context?',
        '       Reply: FAITHFUL or NOT_FAITHFUL + reason."),',
        '    ("human", "Context: {context}',
        '       Question: {question}   Answer: {answer}"),',
        '])',
        '',
        'verdict = (JUDGE_PROMPT | llm_judge | StrOutputParser())',
        '          .invoke({"context": ctx, "question": q, "answer": a})',
        'is_faithful = verdict.upper().startswith("FAITHFUL")',
    ],
    tip=[
        'Faithfulness measures',
        'whether the answer is',
        'grounded in the book.',
        '',
        'It does NOT measure',
        'whether the answer is',
        'factually correct in',
        'the real world.',
        '',
        'Target: >= 90 / 15',
    ],
    fname='images/11_faithfulness_evaluation.png',
)

print('\nAll 11 diagrams generated in images/')
