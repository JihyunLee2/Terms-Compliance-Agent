import streamlit as st
import re
import os
import json
import tempfile
import webbrowser
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
# HITL(Human-in-the-Loop)을 위해 interrupt를 임포트합니다.
from langgraph.types import interrupt 
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()

# LLM 및 임베딩 초기화
embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
llm = ChatUpstage(model="solar-pro2")

MAX_ITERATIONS = 3

class ContractState(TypedDict):
    clause: str
    cleaned_text: str
    unfair_type: str
    related_cases: str
    improvement_proposal: str
    user_feedback: str
    modify_reason: str
    retry_action: str
    session_id: str
    iteration: int
    validation_failed: bool

# --- 원본 노드 함수들 (feedback_node 제외) ---

def load_vectordb():
    # Streamlit은 print() 대신 st.write()나 로깅을 사용하는 것이 좋지만,
    # 여기서는 @st.cache_resource가 관리하므로 콘솔에 한 번만 출력됩니다.
    print("벡터 DB 로드 중...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="contract_laws"
    )
    print("벡터 DB 로드 완료!\n")
    return vectorstore

def is_valid_contract_clause(clause: str) -> tuple[bool, str]:
    clause = clause.strip()
    
    if len(clause) < 20:
        return False, "입력이 너무 짧습니다 (최소 20자 필요)"
    
    contract_keywords = [
        '조항', '조건', '약관', '규정', '제', '항', '조', '자',
        '금지', '가능', '불가', '의무', '책임', '권리', '계약',
        '해지', '중단', '변경', '환불', '배상', '배제', '면책',
        '수수료', '이용료', '결제', '할인', '서비스', '제공',
        '개인정보', '보호', '이용', '관리', '통지', '동의',
        '유효', '기간', '상효', '시행', '효력', '청구', '위반',
        '손해배상', '면책조항', '이용자', '회사', '당사자'
    ]
    
    has_keyword = any(keyword in clause for keyword in contract_keywords)
    
    if not has_keyword:
        return False, "약관 관련 키워드 미검출 (예: 조항, 약관, 조건, 의무 등)"
    
    question_marks = ['?', '？']
    is_question = any(q in clause for q in question_marks)
    
    if is_question:
        return False, "질문 형식으로 보입니다. 약관 조항을 입력해주세요"
    
    return True, "검증 통과"


def retrieve_node(state: ContractState, vectorstore):
    print(f"[노드3] 유사 사례 검색 중...\n")
    
    search_query = f"{state['unfair_type']} {state['cleaned_text']}"
    
    # 사례 검색 (최대 5개 요청)
    results_cases = vectorstore.similarity_search(
        search_query,
        k=5,
        filter={"source_type": "case"}
    )
    
    actual_case_count = len(results_cases)
    print(f"사례 검색: {actual_case_count}개 (요청: 5개)\n")
    
    if actual_case_count == 0:
        print("[경고] 유사 사례 없음. 필터 제거 후 재검색...\n")
        results_cases = vectorstore.similarity_search(search_query, k=5)
        actual_case_count = len(results_cases)
        print(f"필터 제거 후: {actual_case_count}개 검색됨\n")
    
    if actual_case_count == 0:
        print("[경고] 검색 결과 없음\n")
        retrieved_info = "[유사 시정 사례] - 검색 결과 없음"
        return {"related_cases": retrieved_info}
    
    # 법령 검색: 검색된 모든 사례에서 관련법 수집
    related_laws_set = set()
    
    for case in results_cases:
        if case.metadata.get('related_law'):
            related_laws_set.add(case.metadata.get('related_law'))
    
    print(f"수집된 관련법: {related_laws_set}\n")
    
    if related_laws_set:
        combined_search = " ".join(related_laws_set)
        results_laws = vectorstore.similarity_search(
            combined_search,
            k=5,
            filter={"source_type": "law"}
        )
    else:
        # 관련법이 없으면 원본 쿼리로 검색
        results_laws = vectorstore.similarity_search(search_query, k=5, filter={"source_type": "law"})
    
    actual_law_count = len(results_laws)
    print(f"법령 검색: {actual_law_count}개 (요청: 5개)\n")
    
    # 결과 포맷팅
    retrieved_info = f"[유사 시정 사례] ({actual_case_count}개)\n"
    
    for i, doc in enumerate(results_cases, 1):
        date_display = doc.metadata.get('date', 'N/A')
        retrieved_info += f"\n[사례 {i}] ({date_display})\n"
        retrieved_info += f"약관: {doc.page_content.split('결론:')[0].replace('약관: ', '').strip()}\n\n"
        
        if doc.metadata.get('explanation'):
            retrieved_info += f"[시정 요청 사유]\n{doc.metadata.get('explanation')}\n\n"
        
        if doc.metadata.get('conclusion'):
            retrieved_info += f"[최종 결론]\n{doc.metadata.get('conclusion')}\n\n"
        
        if doc.metadata.get('related_law'):
            retrieved_info += f"[관련법]\n{doc.metadata.get('related_law')}\n"
        
        retrieved_info += "-" * 40
    
    retrieved_info += f"\n[관련 법령] ({actual_law_count}개)\n"
    
    for i, doc in enumerate(results_laws, 1):
        retrieved_info += f"\n[법령 {i}]\n{doc.page_content}\n"
    
    print("[노드3] 검색 완료\n")
    
    return {"related_cases": retrieved_info}


def route_feedback(state: ContractState) -> str:
    # (원본 route_feedback 로직과 동일)
    if state.get('validation_failed', False):
        print("\n[라우팅 규칙 적용]")
        print(f"- 조건: validation_failed == True")
        print(f"- 액션: 그래프 즉시 종료")
        print(f"- 상태: 룰베이스 검증 실패\n")
        return "end"
    
    feedback = state.get('user_feedback', '').lower()
    retry_action = state.get('retry_action', '')
    current_iteration = state.get('iteration', 1)
    
    print(f"\n[라우팅 규칙 적용 - 반복 횟수: {current_iteration}/{MAX_ITERATIONS}]")
    
    if feedback == "approved":
        print(f"- 조건: user_feedback == 'approved'")
        print(f"- 액션: 그래프 종료 (결과 저장)")
        print(f"- 상태: 완료\n")
        return "end"
    
    elif feedback == "rejected" and retry_action == "retry":
        print(f"- 조건: user_feedback == 'rejected' AND retry_action == 'retry'")
        print(f"- 액션: generate 노드로 이동 (다른 개선안 생성)")
        print(f"- 상태: 재시도 (새로운 개선안)\n")
        return "generate"
    
    elif feedback == "rejected" and retry_action == "discard":
        print(f"- 조건: user_feedback == 'rejected' AND retry_action == 'discard'")
        print(f"- 액션: 그래프 종료 (폐기)")
        print(f"- 상태: 거절 및 폐기\n")
        return "end"
    
    elif feedback == "modify" and current_iteration < MAX_ITERATIONS:
        next_iteration = current_iteration + 1
        print(f"- 조건: user_feedback == 'modify' AND iteration({current_iteration}) < MAX({MAX_ITERATIONS})")
        print(f"- 액션: generate 노드로 이동 (피드백 반영)")
        print(f"- 상태: 반복 {next_iteration}차 진행\n")
        return "generate"
    
    elif feedback == "modify" and current_iteration >= MAX_ITERATIONS:
        print(f"- 조건: user_feedback == 'modify' AND iteration({current_iteration}) >= MAX({MAX_ITERATIONS})")
        print(f"- 반복 횟수 제한 도달!")
        print(f"- 액션: 그래프 종료 (강제)")
        print(f"- 상태: 반복 제한 도달\n")
        return "end"
    
    else:
        print(f"- 기타 조건")
        print(f"- 액션: 그래프 종료\n")
        return "end"

def clean_text_node(state: ContractState):
    # (원본 clean_text_node 로직과 동일)
    print(f"\n[노드1] Rule-based 검증 + 텍스트 정제\n")
    
    is_valid, validation_msg = is_valid_contract_clause(state['clause'])
    print(f"[Rule-based 검증 결과] {validation_msg}")
    
    if not is_valid:
        print(f"-> API 호출 중단\n")
        return {
            "cleaned_text": "[룰 베이스 거부] 약관 조항이 아님",
            "validation_failed": True
        }
    
    print(f"-> 검증 통과\n")
    
    original_text = state['clause']
    cleaned = original_text
    
    # 불릿 포인트 제거
    cleaned = re.sub(r'^[\s•\-\*]+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    
    # 괄호 번호 제거: (1), (2), (3) 등
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
    
    # 연속된 공백/개행 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    print(f"[정제 전] {len(original_text)}자")
    print(f"{original_text}\n")
    print(f"[정제 후] {len(cleaned)}자")
    print(f"{cleaned}\n")
    
    return {
        "cleaned_text": cleaned,
        "validation_failed": False
    }


def classify_type_node(state: ContractState):
    # (원본 classify_type_node 로직과 동일)
    print(f"[노드2] Solar API - 불공정 유형 분류\n")
    
    prompt = f"""다음 약관 조항의 불공정 유형을 판단하세요:

{state['cleaned_text']}

유형:
1. 서비스 일방적 변경 중단
2. 기한의 이익 상실
3. 고객 이의제기 제한
4. 개별통지 부적절 생략
5. 계약해지 사유 포괄적
6. 기타

위 5가지 유형에 해당하지 않으면 "6. 기타"로 분류하세요.

해당 유형만 출력하세요."""
    
    unfair_type = llm.invoke(prompt).content.strip()
    
    print(f"분류 결과: {unfair_type}\n")
    
    return {"unfair_type": unfair_type}

def generate_proposal_node(state: ContractState):
    # (원본 generate_proposal_node 로직과 동일)
    print(f"[노드4] Solar API - 개선안 생성 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    
    feedback_context = ""
    
    if state.get('modify_reason'):
        feedback_context = f"\n[사용자 피드백]\n{state['modify_reason']}\n위 의견을 반영해 다시 작성하세요.\n"
    
    prompt = f"""당신은 법률 전문가입니다.

[원본 약관 조항]
{state['cleaned_text']}

[불공정 유형]
{state['unfair_type']}

[관련 시정 사례 및 법령]
{state['related_cases']}

{feedback_context}

[작업]
위 정보를 바탕으로 이 약관 조항을 공정한 약관으로 개선하세요.

[중요 규칙]
- 법 근거는 위의 "관련 시정 사례 및 법령"에 명시된 것만 사용하세요.
- 근거 없는 법령이나 조항, 특정 기간(6개월, 90일 등)을 포함하지 마세요.
- 관련 자료에 없는 내용은 생성하지 마세요.

[출력 형식]
1. 개선된 약관 조항
2. 개선 사유 (관련 시정 사례 및 법령에서만 제시)
3. 핵심 변경 사항"""
    
    proposal = llm.invoke(prompt).content
    
    print(f"개선안 생성 완료 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    
    return {"improvement_proposal": proposal}


def process_feedback_node(state: ContractState):
    # (원본 process_feedback_node 로직과 동일)
    feedback = state['user_feedback']
    retry_action = state.get('retry_action', '')
    current_iteration = state.get('iteration', 1)
    
    if feedback == "approved":
        save_result(
            state=state,
            status="approved",
            iteration=current_iteration,
            total_iterations=current_iteration
        )
        print("[노드6] 결과 저장 완료 (수락)")
        print(f"총 {current_iteration}회 반복 후 완료\n")
        return {
            "user_feedback": "approved",
            "retry_action": ""
        }
    
    elif feedback == "rejected":
        if retry_action == "retry":
            new_iteration = current_iteration + 1
            save_result(
                state=state,
                status="rejected_retry",
                iteration=current_iteration
            )
            print(f"[노드6] 거절 기록 (재시도 예정)")
            print(f"-> 반복 {new_iteration}차 진행\n")
            return {
                "user_feedback": "rejected",
                "iteration": new_iteration,
                "retry_action": "retry"
            }
        else: # "discard"
            save_result(
                state=state,
                status="rejected_discard",
                iteration=current_iteration,
                total_iterations=current_iteration
            )
            print(f"[노드6] 결과 저장 완료 (거절 및 폐기)\n")
            return {
                "user_feedback": "rejected",
                "retry_action": "discard"
            }
    
    elif feedback == "modify":
        if current_iteration >= MAX_ITERATIONS:
            save_result(
                state=state,
                status="max_iteration_reached",
                iteration=current_iteration,
                total_iterations=current_iteration,
                modify_reason="반복 횟수 제한 도달"
            )
            print(f"[노드6] 반복 횟수 제한 도달")
            print(f"총 {current_iteration}회 반복 (최대값)\n")
            # 반복 초과 시, modify를 approved로 강제 변환하여 종료
            return {
                "user_feedback": "approved", 
                "retry_action": ""
            }
        
        new_iteration = current_iteration + 1
        save_result(
            state=state,
            status="modify_request",
            iteration=current_iteration,
            modify_reason=state.get('modify_reason', '')
        )
        print(f"[노드6] 수정 요청 저장")
        print(f"-> 반복 {new_iteration}차 진행\n")
        return {
            "user_feedback": "modify",
            "iteration": new_iteration,
            "modify_reason": state.get('modify_reason', ''),
            "retry_action": ""
        }
    
    return {
        "user_feedback": feedback,
        "retry_action": ""
    }

# (원본 feedback_node(input()) 함수는 여기서 삭제됨)

def save_result(state: ContractState, status: str, iteration: int,
                modify_reason: str = "", total_iterations: int = None):
    # (원본 save_result 로직과 동일)
    result = {
        "timestamp": datetime.now().isoformat(),
        "session_id": state['session_id'],
        "status": status,
        "iteration": iteration,
        "total_iterations": total_iterations or iteration,
        "original_clause": state['clause'],
        "cleaned_text": state['cleaned_text'],
        "unfair_type": state['unfair_type'],
        "improvement_proposal": state['improvement_proposal'],
        "modify_reason": modify_reason
    }
    
    filename = f"{status}_data.jsonl"
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

# --- Streamlit에 맞게 수정된 부분 ---

def ui_feedback_node(state: ContractState):
    """
    Streamlit UI에서 피드백을 받기 위해 그래프를 일시 중지(interrupt)합니다.
    이 노드는 'input()' 대신 'interrupt()'를 반환합니다.
    """
    print(f"\n[노드5] UI 피드백 대기 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    print(f"개선안:\n{state['improvement_proposal']}\n")
    # LangGraph를 일시 중지하고 Streamlit UI로 제어권을 넘깁니다.
    return interrupt(state)

@st.cache_resource
def get_app_and_vectorstore():
    """
    Streamlit의 캐시 기능을 사용해 VectorDB와 LangGraph 앱을 한 번만 로드합니다.
    """
    vectorstore = load_vectordb()
    
    graph = StateGraph(ContractState)
    
    # 노드 추가
    graph.add_node("clean", clean_text_node)
    graph.add_node("classify", classify_type_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_proposal_node)
    
    # [중요] 원본 feedback_node 대신 ui_feedback_node(interrupt) 사용
    graph.add_node("feedback", ui_feedback_node) 
    
    graph.add_node("process_feedback", process_feedback_node)
    
    # 진입점 설정
    graph.set_entry_point("clean")
    
    # 엣지 연결 (원본과 동일)
    def route_after_clean(state: ContractState) -> str:
        if state.get('validation_failed', False):
            return "end"
        return "classify"
    
    graph.add_conditional_edges(
        "clean",
        route_after_clean,
        {
            "end": END,
            "classify": "classify"
        }
    )
    
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "feedback") # generate -> feedback(interrupt)
    graph.add_edge("feedback", "process_feedback") # feedback(interrupt) -> process_feedback
    
    graph.add_conditional_edges(
        "process_feedback",
        route_feedback,
        {
            "end": END,
            "generate": "generate"
        }
    )
    
    # 메모리 체커와 함께 앱 컴파일
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    
    return app

# --- Streamlit 챗봇 UI 메인 함수 ---

def main_chatbot_ui():
    st.set_page_config(page_title="법률 약관 검토 챗봇", layout="wide")
    st.title("법률 약관 검토 챗봇")
    st.caption(f"최대 수정 횟수: {MAX_ITERATIONS}회")

    # LangGraph 앱 로드 (캐시됨)
    try:
        app = get_app_and_vectorstore()
    except Exception as e:
        st.error(f"애플리케이션 로드 실패: {e}")
        st.error("Chroma DB 파일('./chroma_db')이 올바르게 위치해 있는지 확인하세요.")
        return

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    # 'hitl_pending': True이면 피드백 버튼을 표시, False이면 채팅 입력을 받음
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = False
    if "current_state" not in st.session_state:
        st.session_state.current_state = {}

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- 1. 피드백 대기 상태일 때 (HITL) ---
    if st.session_state.hitl_pending:
        
        current_iteration = st.session_state.current_state.get('iteration', 1)
        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")

        # 피드백 UI
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("수정 요청 (Modify)")
            modify_reason = st.text_area("수정 요청 사유:", key="modify_reason_input")
            
            # 반복 횟수 체크
            if current_iteration >= MAX_ITERATIONS:
                st.warning(f"반복 횟수 제한({MAX_ITERATIONS}회)에 도달하여 더 이상 수정 요청을 할 수 없습니다.")
                if st.button("현재 개선안 수락 (Approve)", use_container_width=True, type="primary"):
                    # 'modify'가 아닌 'approved'로 피드백을 강제하여 그래프를 종료시킴
                    feedback_input = {
                        "user_feedback": "approved",
                        "modify_reason": "반복 횟수 제한 도달",
                        "retry_action": ""
                    }
                    st.session_state.hitl_pending = False
                    st.session_state.messages.append({"role": "user", "content": "[피드백] 반복 초과로 현재 개선안을 수락합니다."})
                    st.rerun() # UI를 즉시 새로고침하여 다음 invoke 실행

            else:
                if st.button("수정 요청 제출 (Modify)", key="modify_btn", use_container_width=True):
                    if not modify_reason.strip():
                        st.error("수정 요청 사유를 반드시 입력해야 합니다.")
                    else:
                        feedback_input = {
                            "user_feedback": "modify",
                            "modify_reason": modify_reason.strip(),
                            "retry_action": ""
                        }
                        st.session_state.hitl_pending = False
                        st.session_state.messages.append({"role": "user", "content": f"[피드백] 수정 요청:\n{modify_reason.strip()}"})
                        st.rerun()

        with col2:
            st.subheader("수락 또는 거절 (Approve / Reject)")
            if st.button("개선안 수락 (Approve)", key="approve_btn", use_container_width=True):
                feedback_input = {
                    "user_feedback": "approved",
                    "modify_reason": "",
                    "retry_action": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({"role": "user", "content": "[피드백] 개선안을 수락합니다 (완료)."})
                st.rerun()

            if st.button("다른 개선안 생성 (Reject + Retry)", key="retry_btn", use_container_width=True):
                feedback_input = {
                    "user_feedback": "rejected",
                    "retry_action": "retry",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({"role": "user", "content": "[피드백] 거절 (다른 개선안 재시도)."})
                st.rerun()

            if st.button("폐기 (Reject + Discard)", key="discard_btn", use_container_width=True):
                feedback_input = {
                    "user_feedback": "rejected",
                    "retry_action": "discard",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({"role": "user", "content": "[피드백] 거절 (검토 폐기)."})
                st.rerun()
        
        # 피드백 대기 중에는 메인 채팅 입력 비활성화
        st.chat_input("피드백을 먼저 완료해주세요.", disabled=True)

    # --- 2. 일반 입력 대기 상태일 때 ---
    else:
        # (A) 피드백이 방금 제출된 경우 (st.rerun() 직후)
        # 'feedback_input' 변수가 locals()에 있는지 확인
        if "feedback_input" in locals():
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.chat_message("assistant"):
                with st.spinner("피드백을 반영하여 처리 중..."):
                    try:
                        # [중요] None을 전달하여 중단된 지점부터 그래프 재개
                        output = app.invoke(
                            None, # 중단된 지점부터 실행
                            config=config,
                            **feedback_input # UI에서 받은 피드백 전달
                        )
                        st.session_state.current_state = output
                        
                        # 라우팅 결과 확인
                        last_feedback = output.get('user_feedback', '')
                        last_retry = output.get('retry_action', '')

                        # 그래프가 'end'로 라우팅된 경우
                        if last_feedback == "approved" or (last_feedback == "rejected" and last_retry == "discard"):
                            st.markdown("### 검토 완료\n검토가 최종 완료되었습니다. 새로운 약관 조항을 입력하세요.")
                            st.session_state.messages.append({"role": "assistant", "content": "검토가 완료되었습니다. 다음 조항을 입력해주세요."})
                            st.session_state.thread_id = None # 세션 리셋
                            st.session_state.current_state = {}

                        # 그래프가 'generate'로 다시 라우팅된 경우 (modify 또는 retry)
                        else: 
                            st.markdown(f"### 🔄 새로운 개선안 (반복 {output.get('iteration', '?')}/{MAX_ITERATIONS})\n피드백을 반영한 새로운 개선안입니다.")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({"role": "assistant", "content": output['improvement_proposal']})
                            st.session_state.hitl_pending = True # 다시 피드백 대기
                            st.rerun() # 피드백 버튼을 다시 표시하기 위해 rerun

                    except Exception as e:
                        st.error(f"피드백 처리 중 오류 발생: {e}")
                        st.session_state.hitl_pending = False
                        st.session_state.thread_id = None


        # (B) 새로운 약관 조항 입력
        elif prompt := st.chat_input("검토할 약관 조항을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("약관 조항을 분석 중입니다... (1/4)"):
                    try:
                        # 새 세션 시작
                        st.session_state.thread_id = f"session_{datetime.now().timestamp()}"
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        initial_state = {
                            "clause": prompt,
                            "iteration": 1,
                            "session_id": st.session_state.thread_id,
                            "validation_failed": False # 초기화
                        }
                        
                        # LangSmith 트래킹 활성화 (선택 사항)
                        with tracing_v2_enabled():
                            # 그래프 실행 (clean -> classify -> retrieve -> generate -> feedback(interrupt))
                            output = app.invoke(
                                initial_state,
                                config=config
                            )
                        
                        st.session_state.current_state = output
                        
                        # 룰베이스 검증 실패 시 (그래프가 'end'로 즉시 종료됨)
                        if output.get('validation_failed', False):
                            error_msg = f"입력 오류: {output.get('cleaned_text', '알 수 없는 오류')}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.thread_id = None # 세션 리셋
                        
                        # 첫 번째 개선안 생성 완료 (그래프가 'feedback'에서 중지됨)
                        else:
                            st.markdown("### 提案 (첫 번째 개선안)\n제안된 개선안입니다. 검토 후 피드백을 주세요.")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({"role": "assistant", "content": output['improvement_proposal']})
                            st.session_state.hitl_pending = True # 피드백 대기 상태로 전환
                            st.rerun() # UI를 새로고침하여 피드백 버튼 표시

                    except Exception as e:
                        st.error(f"약관 검토 중 오류 발생: {e}")
                        import traceback
                        st.exception(traceback.format_exc())
                        st.session_state.thread_id = None


if __name__ == "__main__":
    # 원본 main() 대신 Streamlit UI 함수를 실행합니다.
    main_chatbot_ui()