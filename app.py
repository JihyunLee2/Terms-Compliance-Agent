# app.py - 최종 통합 버전

# 1. Rule-based Input 검증 (Rule-based 검증 실패 시 즉시 종료)
# 2. 메타데이터 필터 (날짜 기반)
# 3. HITL (rejected + retry/discard)
# 4. 반복 횟수 제한
# 5. LangSmith 트래킹

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
from langgraph.types import interrupt
from langchain_core.tracers.context import tracing_v2_enabled

load_dotenv()

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
def feedback_node(state: ContractState):
    current_iteration = state.get('iteration', 1)
    
    print(f"\n{'='*60}")
    print(f"[생성된 개선안 (반복: {current_iteration}/{MAX_ITERATIONS})]")
    print(f"{'='*60}\n")
    print(f"{state['improvement_proposal']}\n")
    print(f"{'='*60}\n")
    
    print("평가 옵션:")
    print("1. approved - 수락 (완료)")
    print("2. rejected - 거절")
    print("3. modify - 수정 요청")
    
    if current_iteration >= MAX_ITERATIONS:
        print(f"\n알림: 이번이 마지막 수정 요청입니다 (반복 {MAX_ITERATIONS}차 제한)")
        print(f"알림: 다음 선택 후에는 반드시 수락 또는 거절해야 합니다\n")
    
    while True:
        feedback = input("선택 (approved/rejected/modify): ").strip().lower()
        
        if feedback == "rejected":
            print("\n거절 후 다음 작업을 선택하세요:")
            print("1. retry - 다른 개선안 생성 (재시도)")
            print("2. discard - 폐기 (종료)\n")
            
            retry_action = input("선택 (retry/discard): ").strip().lower()
            
            if retry_action not in ["retry", "discard"]:
                print("경고: 잘못된 입력. 기본값 'discard' 적용\n")
                retry_action = "discard"
            
            return {
                "user_feedback": "rejected",
                "retry_action": retry_action,
                "modify_reason": ""
            }
        
        elif feedback == "modify":
            if current_iteration >= MAX_ITERATIONS:
                print(f"\n반복 횟수 제한 도달 ({current_iteration}/{MAX_ITERATIONS})")
                print("더 이상 수정 요청을 할 수 없습니다.")
                print("다음 중 선택하세요:")
                print("1. approved - 현재 개선안 수락")
                print("2. rejected - 거절 및 폐기\n")
                
                final_choice = input("선택 (approved/rejected): ").strip().lower()
                
                if final_choice == "rejected":
                    return {
                        "user_feedback": "rejected",
                        "retry_action": "discard",
                        "modify_reason": ""
                    }
                else:
                    return {
                        "user_feedback": "approved",
                        "modify_reason": "",
                        "retry_action": ""
                    }
            
            print("\n수정 요청 사유를 입력하세요:")
            print("(예: 더 구체적인 기준이 필요, 다른 법령 적용 등)\n")
            
            modify_reason = input(">>> ").strip()
            
            if not modify_reason:
                print("경고: 수정 사유를 입력해주세요\n")
                continue
            
            return {
                "user_feedback": "modify",
                "modify_reason": modify_reason,
                "retry_action": ""
            }
        
        elif feedback == "approved":
            return {
                "user_feedback": "approved",
                "modify_reason": "",
                "retry_action": ""
            }
        
        else:
            print("경고: 잘못된 입력입니다. (approved/rejected/modify만 입력하세요)\n")


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

def build_graph(vectorstore):
    graph = StateGraph(ContractState)
    
    graph.add_node("clean", clean_text_node)
    graph.add_node("classify", classify_type_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_proposal_node)
    graph.add_node("feedback", feedback_node)
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

def main():
    vectorstore = load_vectordb()
    app = build_graph(vectorstore)
    
    while True:
        print("\n" + "="*60)
        clause = input("검토할 약관 조항을 입력하세요 (종료: 'quit'): ").strip()
        
        if clause.lower() == 'quit':
            print("프로그램 종료")
            break
        
        if not clause:
            print("약관 조항을 입력해주세요")
            continue
        
        session_id = f"session_{datetime.now().timestamp()}"
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            initial_state = {
                "clause": clause,
                "cleaned_text": "",
                "unfair_type": "",
                "related_cases": "",
                "improvement_proposal": "",
                "user_feedback": "",
                "modify_reason": "",
                "retry_action": "",
                "session_id": session_id,
                "iteration": 1,
                "validation_failed": False
            }
            
            with tracing_v2_enabled():
                output = app.invoke(
                    initial_state,
                    config=config
                )
        
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
