import os
import json
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

load_dotenv()

INDEX_NAME = "wellbeing-coach-rag"

SYSTEM_PROMPT = '''You are a Wellbeing Coach who helps DanceSport athletes answer questions on competition prep and performance readiness. Answer using ONLY the provided context. If you cannot find the answer, say 'I don\'t have that in my knowledge base.' Do NOT hallucinate.

You are top expert in the domains at the intersection of DanceSport, High Performance, Wellbeing. Accuracy beats approval. Blunt, argumentative. No disclaimers or praise. Lead with counterarguments. Don\'t capitulate without new evidence.

TAG every claim: [KNOWN] training fact · [COMPUTED] calculated · [INFERRED] deduction · [COMMON] standard field knowledge · [FRAME] symbolic system, coherent ≠ real · [GUESS] no basis. No untagged disease, statute, citation, or named entity.

FRAME→REALITY FORBIDDEN: Don\'t translate symbolic frames (astrology, typologies) into real-world claims (medicine, law, finance) without flagging the translation; conclusion stays in source frame.

CONFIDENCE: HIGH ≥80% · MED 50–80% · LOW 20–50% · VERY LOW <20% · UNKNOWN. [FRAME] real-world and [GUESS] cap at LOW.

DON\'T KNOW: First line "I don\'t know." Don\'t bury, don\'t fabricate.

ANTI-SYCOPHANCY red flags: unusually elegant; one pattern explains everything; agreed after pushback without evidence; specifics for unearned authority. Fire → cut specifics, add [GUESS], or "I don\'t know."

POST-HOC: Would the frame predict this without knowing the outcome? If no: [INFERRED, post-hoc], accommodates, doesn\'t predict.

Never fabricate citations. Revise openly if holding a position for consistency. Append "[RULES I BROKE]: which, where, why."

CITATION FORMAT: After every claim, add an inline citation: [Dance To Your Maximum, Chapter X-Y, Page Z–W]
Use the chapter and page metadata from the provided context documents.'''

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify the user's DanceSport query into routing categories.

part_scope options:
- competition_day: competition day stress, morning routine, warm-up, floor performance
- season: season planning, choreography, exercises, training load
- career: long-term goals, career arc, identity, calling
- general: spans multiple parts or unclear

section_type options:
- test: specific test, questionnaire, or self-assessment
- form: evaluation form or worksheet
- appendix: appendix content
- prose: regular explanatory content
- general: unclear

Return valid JSON only. Example: {{"part_scope": "competition_day", "section_type": "prose"}}"""),
    ("human", "Query: {question}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context from knowledge base:\n\n{context}\n\nQuestion: {question}"),
])


class RAGState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    route: dict
    documents: List[Document]
    answer: str


def _get_components():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_answer = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
    return vectorstore, llm, llm_answer


def build_filter(route: dict) -> dict:
    filters = {}
    part_scope = route.get("part_scope", "general")
    section_type = route.get("section_type", "general")
    if part_scope and part_scope != "general":
        filters["part_scope"] = {"$eq": part_scope}
    if section_type and section_type not in {"prose", "general"}:
        filters["section_type"] = {"$in": [section_type]}
    return filters


def format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        meta = doc.metadata
        chapter = meta.get("chapter", "N/A")
        page_start = meta.get("page_start", "")
        page_end = meta.get("page_end", "")
        if page_start and page_end and str(page_start) != str(page_end):
            cite = f"[Dance To Your Maximum, Chapter {chapter}, Page {page_start}–{page_end}]"
        elif page_start:
            cite = f"[Dance To Your Maximum, Chapter {chapter}, Page {page_start}]"
        else:
            cite = f"[Dance To Your Maximum, Chapter {chapter}]"
        parts.append(f"{doc.page_content}\n\nSource: {cite}")
    return "\n\n---\n\n".join(parts)


def build_rag_graph():
    vectorstore, llm, llm_answer = _get_components()

    def route_node(state: RAGState) -> dict:
        chain = ROUTER_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"question": state["question"]})
        try:
            route = json.loads(result)
        except json.JSONDecodeError:
            route = {"part_scope": "general", "section_type": "general"}
        return {"route": route}

    def retrieve_node(state: RAGState) -> dict:
        route = state.get("route", {})
        filter_dict = build_filter(route)
        if filter_dict:
            docs = vectorstore.similarity_search(state["question"], k=4, filter=filter_dict)
            if not docs:
                docs = vectorstore.similarity_search(state["question"], k=6)
        else:
            docs = vectorstore.similarity_search(state["question"], k=6)
        return {"documents": docs}

    def generate_node(state: RAGState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"answer": "I don't have that in my knowledge base."}
        context = format_context(docs)
        chain = RAG_PROMPT | llm_answer | StrOutputParser()
        answer = chain.invoke({"context": context, "question": state["question"]})
        return {"answer": answer}

    graph = StateGraph(RAGState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_rag_app = None


def get_rag_app():
    global _rag_app
    if _rag_app is None:
        _rag_app = build_rag_graph()
    return _rag_app


def ask(question: str, chat_history: list = None) -> str:
    app = get_rag_app()
    result = app.invoke({
        "question": question,
        "chat_history": chat_history or [],
        "route": {},
        "documents": [],
        "answer": "",
    })
    return result["answer"]
