from config2 import *
import streamlit as st
import re
import os
import json
from datetime import datetime
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt 
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()

# LLM 및 임베딩 초기화
embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
llm = ChatUpstage(model=LLM_MODEL)

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
    retrieved_cases_metadata: List[dict]
    retrieved_laws_metadata: List[dict]
    similarity_threshold: float

def load_vectordb():
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
        '유효', '기간', '상호', '시행', '효력', '청구', '위반',
        '손해배상', '면책조항', '이용자', '회사', '당사자'
    ]
    
    has_keyword = any(keyword in clause for keyword in contract_keywords)
    
    if not has_keyword:
        return False, "약관 관련 키워드 미검출 (예: 조항, 약관, 조건, 의무 등)"
    
    question_marks = ['?', '?']
    is_question = any(q in clause for q in question_marks)
    
    if is_question:
        return False, "질문 형식으로 보입니다. 약관 조항을 입력해주세요"
    
    return True, "검증 통과"

def clean_text_node(state: ContractState):
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
    
    cleaned = re.sub(r'^[\s•\-\*]+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
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

def retrieve_node(state: ContractState, vectorstore):
    
    current_threshold = state.get('similarity_threshold', SIMILARITY_THRESHOLD)
    print(f"[노드3] 검색 (임계값: {current_threshold:.0%})")
    
    search_query = f"{state['unfair_type']} {state['cleaned_text']}"
    
    # 1. 사례 검색 (유사도 점수 포함)
    # 이 함수는 (doc, similarity_score) 튜플을 반환합니다. (1.0이 100% 유사)
    results_cases_with_scores = vectorstore.similarity_search_with_relevance_scores(
        search_query, 
        k=SEARCH_TOP_K_CASES, 
        filter={"source_type": "case"}
    )
    
    filtered_cases_meta = []
    
    for i, (doc, similarity_score) in enumerate(results_cases_with_scores, 1):
        
        # similarity_score (예: 0.75)를 current_threshold (예: 0.70)와 직접 비교
        if similarity_score >= current_threshold:
            print(f"  ✓ 사례 통과 (유사도 {similarity_score:.1%})") # 디버깅용 로그
            filtered_cases_meta.append({
                "index": i,
                "similarity": similarity_score, # 계산이 아닌, 반환된 점수 그대로 사용
                "content": doc.page_content,
                "date": doc.metadata.get('date', 'N/A'),
                "case_type": doc.metadata.get('case_type', ''),
                "explanation": doc.metadata.get('explanation', ''),
                "conclusion": doc.metadata.get('conclusion', ''),
                "related_law": doc.metadata.get('related_law', '')
            })
        else:
             print(f"  ✗ 사례 필터됨 (유사도 {similarity_score:.1%})") # 디버깅용 로그

    # 표시 개수 제한
    final_cases_meta = filtered_cases_meta[:MAX_DISPLAY_CASES]
    
    # 2. 법령 검색
    law_query = " ".join([c['related_law'] for c in final_cases_meta if c['related_law']])
    if not law_query: law_query = search_query
        
    results_laws_with_scores = vectorstore.similarity_search_with_relevance_scores(
        law_query, 
        k=SEARCH_TOP_K_LAWS, 
        filter={"source_type": "law"}
    )
    
    final_laws_meta = []
    for i, (doc, similarity_score) in enumerate(results_laws_with_scores, 1):
        if similarity_score >= current_threshold:
            final_laws_meta.append({
                "index": i,
                "similarity": similarity_score,
                "content": doc.page_content
            })
    final_laws_meta = final_laws_meta[:MAX_DISPLAY_LAWS]

    # 3. LLM 프롬프트용 텍스트 생성
    retrieved_text = f"[유사 시정 사례] ({len(final_cases_meta)}건, 임계점: {current_threshold:.0%})\n"
    
    for c in final_cases_meta:
        retrieved_text += f"\n- 사례{c['index']} (유사도 {c['similarity']:.1%}): {c['content']}\n"
        if c['related_law']:
            retrieved_text += f"  (관련법: {c['related_law']})\n"
    
    retrieved_text += f"\n[관련 법령] ({len(final_laws_meta)}건)\n"
    for l in final_laws_meta:
        retrieved_text += f"- 법령{l['index']} (유사도 {l['similarity']:.1%}): {l['content']}\n"

    return {
        "related_cases": retrieved_text,
        "retrieved_cases_metadata": final_cases_meta,
        "retrieved_laws_metadata": final_laws_meta
    }

def generate_proposal_node(state: ContractState):
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

def ui_feedback_node(state: ContractState):
    print(f"\n[노드5] UI 피드백 대기 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")
    print(f"개선안:\n{state['improvement_proposal']}\n")
    
    if state.get('retrieved_cases_metadata'):
        print(f"참고한 사례 수: {len(state['retrieved_cases_metadata'])}개")
        for case in state['retrieved_cases_metadata']:
            print(f"  - 사례 {case['index']}: 유사도 {case['similarity']:.2%}")
    
    return interrupt(state)

def process_feedback_node(state: ContractState):
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
        else:
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

def route_feedback(state: ContractState) -> str:
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

def save_result(state: ContractState, status: str, iteration: int,
                modify_reason: str = "", total_iterations: int = None):
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

@st.cache_resource
def get_app_and_vectorstore():
    vectorstore = load_vectordb()
    
    graph = StateGraph(ContractState)
    
    graph.add_node("clean", clean_text_node)
    graph.add_node("classify", classify_type_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_proposal_node)
    graph.add_node("feedback", ui_feedback_node)
    graph.add_node("process_feedback", process_feedback_node)
    
    graph.set_entry_point("clean")
    
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
    graph.add_edge("generate", "feedback")
    graph.add_edge("feedback", "process_feedback")
    
    graph.add_conditional_edges(
        "process_feedback",
        route_feedback,
        {
            "end": END,
            "generate": "generate"
        }
    )
    
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    
    return app

def main_chatbot_ui():
    st.set_page_config(page_title="법률 약관 검토 챗봇", layout="wide")
    st.title("법률 약관 검토 챗봇")
    
    with st.sidebar:
        st.header("⚙️ 검색 설정")
        similarity_threshold_percent = st.slider(
            "유사도 임계값 (%)",
            min_value=0,
            max_value=100,
            value=int(SIMILARITY_THRESHOLD * 100), # config 기본값 사용
            step=5,
            format="%d%%"
        )
        current_threshold_value = similarity_threshold_percent / 100.0
        st.caption(f"현재 설정: {similarity_threshold_percent}% 이상")
        st.divider()

    try:
        app = get_app_and_vectorstore()
    except Exception as e:
        st.error(f"애플리케이션 로드 실패: {e}")
        st.error("Chroma DB 파일('./chroma_db')이 올바르게 위치해 있는지 확인하세요.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = False
    if "current_state" not in st.session_state:
        st.session_state.current_state = {}
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.hitl_pending:
        current_iteration = st.session_state.current_state.get('iteration', 1)
        used_threshold = st.session_state.current_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")

        if SHOW_RETRIEVED_CASES:
            with st.expander("참고한 유사 사례 보기 (비교용)", expanded=True):
                cases = st.session_state.current_state.get('retrieved_cases_metadata', [])
                
                if cases:
                    st.caption(f"총 {len(cases)}개 사례 (유사도 {used_threshold:.0%} 이상)")
                    
                    for case in cases:
                        similarity = case['similarity']
                        
                        if similarity >= 0.9:
                            color = "🟢"
                        elif similarity >= 0.8:
                            color = "🟡"
                        else:
                            color = "🟠"
                        
                        st.markdown(f"### {color} 사례 {case['index']} - 유사도: {similarity:.1%}")
                        st.caption(f"📅 {case['date']} | 유형: {case['case_type']}")
                        
                        with st.container():
                            st.markdown("**약관 조항:**")
                            st.info(case['content'].split('결론:')[0].replace('약관: ', '').strip())
                            
                            if case['explanation']:
                                st.markdown("**시정 요청 사유:**")
                                st.warning(case['explanation'])
                            
                            if case['conclusion']:
                                st.markdown("**최종 결론:**")
                                st.success(case['conclusion'])
                            
                            if case['related_law']:
                                st.caption(f"🔗 관련법: {case['related_law']}")
                        
                        st.divider()
                else:
                    st.warning("검색된 사례가 없습니다.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("수정 요청 (Modify)")
            modify_reason = st.text_area("수정 요청 사유:", key="modify_reason_input")
            
            if current_iteration >= MAX_ITERATIONS:
                st.warning(f"반복 횟수 제한({MAX_ITERATIONS}회)에 도달하여 더 이상 수정 요청을 할 수 없습니다.")
                if st.button("현재 개선안 수락 (Approve)", use_container_width=True, type="primary"):
                    st.session_state.pending_feedback = {
                        "user_feedback": "approved",
                        "modify_reason": "반복 횟수 제한 도달",
                        "retry_action": ""
                    }
                    st.session_state.hitl_pending = False
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "[피드백] 반복 초과로 현재 개선안을 수락합니다."
                    })
                    st.rerun()
            else:
                if st.button("수정 요청 제출 (Modify)", key="modify_btn", use_container_width=True):
                    if not modify_reason.strip():
                        st.error("수정 요청 사유를 반드시 입력해야 합니다.")
                    else:
                        st.session_state.pending_feedback = {
                            "user_feedback": "modify",
                            "modify_reason": modify_reason.strip(),
                            "retry_action": ""
                        }
                        st.session_state.hitl_pending = False
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": f"[피드백] 수정 요청:\n{modify_reason.strip()}"
                        })
                        st.rerun()

        with col2:
            st.subheader("수락 또는 거절 (Approve / Reject)")
            if st.button("개선안 수락 (Approve)", key="approve_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "approved",
                    "modify_reason": "",
                    "retry_action": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 개선안을 수락합니다 (완료)."
                })
                st.rerun()

            if st.button("다른 개선안 생성 (Reject + Retry)", key="retry_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "rejected",
                    "retry_action": "retry",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 거절 (다른 개선안 재시도)."
                })
                st.rerun()

            if st.button("폐기 (Reject + Discard)", key="discard_btn", use_container_width=True):
                st.session_state.pending_feedback = {
                    "user_feedback": "rejected",
                    "retry_action": "discard",
                    "modify_reason": ""
                }
                st.session_state.hitl_pending = False
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "[피드백] 거절 (검토 폐기)."
                })
                st.rerun()
        
        st.chat_input("피드백을 먼저 완료해주세요.", disabled=True)

    else:
        if st.session_state.pending_feedback is not None:
            feedback_input = st.session_state.pending_feedback
            st.session_state.pending_feedback = None
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.chat_message("assistant"):
                with st.spinner("피드백을 반영하여 처리 중..."):
                    try:
                        output = app.invoke(None, config=config, **feedback_input)
                        st.session_state.current_state = output
                        
                        last_feedback = output.get('user_feedback', '')
                        last_retry = output.get('retry_action', '')

                        if last_feedback == "approved" or (last_feedback == "rejected" and last_retry == "discard"):
                            st.markdown("### 검토 완료\n검토가 최종 완료되었습니다.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "검토가 완료되었습니다."
                            })
                            st.session_state.thread_id = None
                            st.session_state.current_state = {}
                        else:
                            st.markdown(f"### 🔄 새로운 개선안 (반복 {output.get('iteration', '?')}/{MAX_ITERATIONS})")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"피드백 처리 중 오류 발생: {e}")
                        st.session_state.hitl_pending = False
                        st.session_state.thread_id = None
                        st.session_state.current_state = {}

        elif prompt := st.chat_input("검토할 약관 조항을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("약관 조항을 분석 중입니다..."):
                    try:
                        st.session_state.thread_id = f"session_{datetime.now().timestamp()}"
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        initial_state = {
                            "clause": prompt,
                            "iteration": 1,
                            "session_id": st.session_state.thread_id,
                            "validation_failed": False,
                            "retrieved_cases_metadata": [],
                            "retrieved_laws_metadata": [],
                            "similarity_threshold": current_threshold_value
                        }
                        
                        with tracing_v2_enabled():
                            output = app.invoke(initial_state, config=config)
                        
                        st.session_state.current_state = output
                        
                        if output.get('validation_failed', False):
                            error_msg = f"입력 오류: {output.get('cleaned_text', '알 수 없는 오류')}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.thread_id = None
                        else:
                            st.markdown("### 제안 (첫 번째 개선안)")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"약관 검토 중 오류 발생: {e}")
                        import traceback
                        st.exception(traceback.format_exc())
                        st.session_state.thread_id = None
                        st.session_state.hitl_pending = False
                        st.session_state.current_state = {}


if __name__ == "__main__":
    main_chatbot_ui()