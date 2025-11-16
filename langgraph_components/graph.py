from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
import streamlit as st # @st.cache_resource 사용

# 모듈에서 설정, 상태, 노드 로드
from config2 import embeddings, CHROMA_DB_PATH, COLLECTION_NAME, CHROMA_COLLECTION_METADATA
from .state import ContractState  # '.'은 '현재 폴더'를 의미
from . import nodes               # '.'은 '현재 폴더'를 의미
import os


# --- 라우팅 함수 ---

def route_after_clean(state: ContractState) -> str:
    """[라우터 1] 룰 기반 검증 결과에 따라 분기"""
    if state.get('validation_failed', False):
        print("[라우팅] 룰 기반 검증 실패 -> END")
        return "end"
    print("[라우팅] 룰 기반 검증 통과 -> fairness_classification")
    return "fairness_classification"

def route_after_fairness(state: ContractState) -> str:
    """[라우터 2] 공정/불공정 분류 결과에 따라 분기"""
    classification = state.get('fairness_label', '') # app.py 호환
    
    if classification == "":
        print("[라우팅] 공정/불공정 분류 (반복 필요) -> fairness_classification")
        return "fairness_classification"  # 재분류
    elif classification == "공정":
        print("[라우팅] 공정/불공정 분류 (공정) -> -> retrieve")
        return "retrieve" # RAG 검색을 위해 retrieve로
    elif classification == "불공정":
        print("[라우팅] 공정/불공정 분류 (불공정) -> classify")
        return "classify"
    else:
        print("[라우팅] 공정/불공정 분류 (알 수 없음) -> END")
        return "end"
    
def route_after_retrieve(state: ContractState) -> str:
    """[라우터 2.5] RAG 검색 후, 공정/불공정에 따라 생성 노드 분기"""
    if state.get('fairness_label') == "공정":
        print("-> 액션: '공정' 리포트 생성 (generate_fair_report)")
        return "generate_fair_report"
    else:
        print("-> 액션: '불공정' 개선안 생성 (generate_proposal)")
        return "generate_proposal"

def route_feedback(state: ContractState) -> str:
    """[라우터 3] 피드백 처리 결과에 따라 분기"""
    feedback = state.get('user_feedback', '').lower()
    retry_action = state.get('retry_action', '')
    
    if feedback == "approved" or (feedback == "rejected" and retry_action == "discard"):
        print("-> 액션: 그래프 종료")
        return "end"
    
    elif (feedback == "rejected" and retry_action == "retry") or (feedback == "modify"):
        print("-> 액션: 개선안 재생성 (generate)")
        return "generate"
    
    else:
        print("-> 액T션: (기타) 그래프 종료")
        return "end"


# --- 그래프 생성 ---

def create_graph(vectorstore: Chroma):
    """LangGraph 앱을 생성, 컴파일하고 반환합니다."""
    
    graph = StateGraph(ContractState)
    
    # 노드 추가
    graph.add_node("clean", nodes.clean_text_node)
    graph.add_node("fairness_classification", nodes.fairness_classify_node)
    graph.add_node("classify", nodes.classify_type_node)
    graph.add_node("retrieve", partial(nodes.retrieve_node, vectorstore=vectorstore))
    graph.add_node("generate_proposal", nodes.generate_proposal_node)
    graph.add_node("generate_fair_report", nodes.generate_fair_report_node) # <-- 새 노드 추가
    graph.add_node("feedback", nodes.interrupt_for_feedback_node) # HITL 중지
    graph.add_node("process_feedback", nodes.process_feedback_node)
    
    # 엣지(흐름) 연결
    graph.set_entry_point("clean")
    
    graph.add_conditional_edges(
        "clean",
        route_after_clean,
        {"end": END, "fairness_classification": "fairness_classification"}
    )
    
    graph.add_conditional_edges(
        "fairness_classification",
        route_after_fairness,
        {
            "fairness_classification": "fairness_classification", # 반복
            "classify": "classify",                         # 불공정
            "retrieve": "retrieve",                         # '공정'일 때 RAG로
            "end": END                                      # 알 수 없음
        }
    )
    
    graph.add_edge("classify", "retrieve")
    
    # 'retrieve' 노드 실행 이후에 'route_after_retrieve' 함수를 조건으로 사용
    graph.add_conditional_edges(
        "retrieve", # <-- 'route_after_retrieve' 노드 대신 'retrieve' 노드를 기준으로 분기
        route_after_retrieve, # <-- 이 함수가 라우터(조건) 역할을 함
        {
            "generate_fair_report": "generate_fair_report",
            "generate_proposal": "generate_proposal"
        }
    )

    graph.add_edge("generate_fair_report", END) # '공정' 리포트는 피드백 없이 종료
    graph.add_edge("generate_proposal", "feedback") # '불공정' 개선안은 피드백 대기
    graph.add_edge("feedback", "process_feedback") # 피드백 받아서 재개
    graph.add_conditional_edges(
        "process_feedback",
        route_feedback,
        # 'generate' -> 'generate_proposal'로 이름 변경
        {"end": END, "generate": "generate_proposal"}
    )
    
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    print("\n--- LangGraph 컴파일 완료 ---")
    return app


# --- 앱 로드 유틸리티 (app.py에서 분리) ---

def load_vectordb():
    """Vectorstore를 로드합니다."""
    print("벡터 DB 로드 중...")
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"Chroma DB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata=CHROMA_COLLECTION_METADATA
    )
    print("벡터 DB 로드 완료!\n")
    return vectorstore

@st.cache_resource
def load_app_safe():
    """
    Streamlit 캐시를 사용하여 LangGraph 앱과 Vectorstore를 로드합니다.
    """
    try:
        vectorstore = load_vectordb()
        app = create_graph(vectorstore)
        return app, vectorstore
    except Exception as e:
        st.error(f"애플리케이션 로드 실패: {e}")
        st.error(f"Chroma DB 파일('{CHROMA_DB_PATH}') 또는 설정을 확인하세요.")
        return None, None