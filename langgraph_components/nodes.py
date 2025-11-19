import re
import os
import json
from datetime import datetime
from collections import Counter
from langgraph.types import interrupt

from langchain_community.vectorstores import Chroma
from config2 import *
from .state import ContractState
from .prompts import (
    ACTIVE_FAIRNESS_CLASSIFY_PROMPT, 
    ACTIVE_UNFAIR_TYPE_PROMPT, 
    ACTIVE_GENERATE_PROPOSAL_PROMPT
)

from utils import (
    format_rag_results, 
    save_result,          
    is_valid_contract_clause, 
    clean_clause_text,  
    LAW_FILENAME_MAP      
)

# law_priority 메타데이터 수동 정의 ==============================================================
def assign_law_priority(file_path: str) -> int:
    if "약관법" in file_path:
        return 1
    elif "전자금융거래법" in file_path:
        return 2
    elif "금융소비자보호" in file_path and "시행령" not in file_path and "감독규정" not in file_path:
        return 3
    elif "시행령" in file_path:
        return 4
    elif "감독규정" in file_path:
        return 5
    elif "심사지침" in file_path:
        return 6
    return 99  # 기타
# law_priority 메타데이터 수동 정의 ==============================================================

# --- 그래프 노드 ---

def clean_text_node(state: ContractState):
    """[노드 1] 룰 기반 검증 및 텍스트 정제"""
    print(f"\n[노드1] Rule-based 검증 + 텍스트 정제\n")
    is_valid, validation_msg = is_valid_contract_clause(state['clause'])
    print(f"[Rule-based 검증 결과] {validation_msg}")
    
    if not is_valid:
        print(f"-> API 호출 중단\n")
        return {"cleaned_text": f"[룰 베이스 거부] {validation_msg}", "validation_failed": True}
    
    print(f"-> 검증 통과\n")
    cleaned = clean_clause_text(state['clause'])

    print(f"[정제 전] {len(state['clause'])}자\n[정제 후] {len(cleaned)}자\n{cleaned}\n")
    return {
        "cleaned_text": cleaned, 
        "validation_failed": False,
        "fairness_retry_count": 0,  # app.py 호환성을 위해 fairness_retry_count 초기화
    }

def fairness_classify_node(state: ContractState):
    """[노드 2] 공정/불공정 분류 (내부 반복 및 Fast Path 적용)"""
    cleaned_text = state.get('cleaned_text', '').strip()
    if not cleaned_text:
        print("[노드2] 검사 대상 약관 텍스트가 없습니다. 분류 건너뜀")
        return {}

    print(f"\n[노드2] 공정/불공정 분류 시작 (최대 {FAIRNESS_MAX_ITERATIONS}회, 임계값 {CONFIDENCE_THRESHOLD})\n")
    
    results_history = []
    
    # [반복 로직] 노드 내부에서 최대 횟수만큼 시도
    for iteration in range(1, FAIRNESS_MAX_ITERATIONS + 1):
        prompt = ACTIVE_FAIRNESS_CLASSIFY_PROMPT.format(cleaned_text=cleaned_text)
        
        try:
            response = llm.invoke(prompt).content.strip()
            
            # --- 파싱 로직 ---
            lines = response.split("\n")
            classification_text = lines[0].strip()
            # 신뢰도가 없거나 파싱 실패 시 0.0 처리
            confidence = float(lines[1].strip()) if len(lines) > 1 else 0.0
            
            # 라벨 정규화 ('공정'/'불공정' 키워드 포함 여부 확인)
            label = 'N/A'
            if '공정' in classification_text and '불공정' not in classification_text:
                label = '공정'
            elif '불공정' in classification_text:
                label = '불공정'
                
            if label == 'N/A':
                print(f"  [경고] {iteration}차 시도: 식별 불가 ({classification_text})")
                continue

            # 결과 기록
            results_history.append((label, confidence))
            print(f"  [{iteration}차] 결과: {label}, 신뢰도: {confidence:.2f}")

            # --- 1. Fast Path (즉시 확정) ---
            # 임계값 이상이면 루프 즉시 중단하고 결과 반환
            if confidence >= CONFIDENCE_THRESHOLD:
                print(f"  [Fast Path] {iteration}차에서 기준 충족. 즉시 종료.")
                return {
                    "fairness_label": label,
                    "fairness_confidence": confidence,
                    "fairness_retry_count": iteration,
                    "results_history": results_history
                }

        except Exception as e:
            print(f"  [오류] {iteration}차 시도 중 예외: {e}")
            results_history.append(('error', 0.0))

    # --- 2. Fallback (반복 종료 후 결정) ---
    # 모든 시도가 임계값 미만이거나 실패한 경우
    print(f"  [Fallback] {FAIRNESS_MAX_ITERATIONS}회 모두 임계값 미달. 최고 신뢰도 결과 선택.")

    valid_results = [(l, c) for l, c in results_history if l not in ['N/A', 'error'] and c > 0]

    if not valid_results:
        final_label = "분류 실패"
        final_conf = 0.0
    else:
        # 신뢰도가 가장 높은 결과 선택 (제공된 코드 로직 반영)
        final_label, final_conf = max(valid_results, key=lambda x: x[1])

    print(f"[최종 판단] 분류: {final_label}, 확신도: {final_conf:.2f}")

    return {
        "fairness_label": final_label,
        "fairness_confidence": final_conf,
        "fairness_retry_count": FAIRNESS_MAX_ITERATIONS,
        "results_history": results_history
    }

def classify_type_node(state: ContractState, max_retry=3):
    """[노드 3] 불공정 유형 분류"""
    print(f"\n[노드3] Solar API - 불공정 유형 분류\n")
    
    if not state.get('cleaned_text', ''):
        print("약관 텍스트가 없어 프롬프트 호출 건너뜀")
        return {}
    
    prompt = ACTIVE_UNFAIR_TYPE_PROMPT.format(cleaned_text=state['cleaned_text'])
    
    retry_count = 0
    while retry_count < max_retry:
        try:
            response = llm.invoke(prompt)
            unfair_type = response.content.strip()
            if unfair_type:
                print(f"불공정 유형 분류 결과: {unfair_type}\n")
                return {"unfair_type": unfair_type}
        except Exception as e:
            print(f"llm.invoke 호출 중 예외 발생: {e}, 재시도 {retry_count + 1}회차")
        retry_count += 1
        
    print("최대 재시도 횟수 초과로 불공정 유형 분류 실패")
    return {"unfair_type": "분류 실패"}

def retrieve_node(state: ContractState, vectorstore: Chroma):
    """[노드 4] RAG - 유사 사례 및 법령 검색"""
    current_threshold = state.get('similarity_threshold', SIMILARITY_THRESHOLD)  #
    fairness_label = state.get('fairness_label', '') # '공정'/'불공정' 라벨 확인    #
    print(f"[노드4] 검색 (임계값: {current_threshold:.0%}), fairness_label:{fairness_label}") # 
    
    # --- 쿼리 분기 ---
    if fairness_label == "공정":
        # '공정'일 땐 '불공정 유형'이 없으므로 텍스트만으로 검색 (대조 검색)
        search_query = state['cleaned_text']
        print(f"[노드4] 검색 ('공정' 대조 전략. 임계값: {current_threshold:.0%})")
    else:
        # '불공정'일 땐 유형을 포함하여 검색 (기존 방식)
        search_query = f"{state['unfair_type']} {state['cleaned_text']}"
        print(f"[노드4] 검색 ('불공정' 전략. 임계값: {current_threshold:.0%})")
    # --------------
    
    print(f"\n[노드4] 유사 사례 검색 중...\n")
    
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
    law_query = search_query
        
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
                "content": doc.page_content,
                "metadata": doc.metadata
            })
    final_laws_meta = final_laws_meta[:MAX_DISPLAY_LAWS]
    
    # 3. LLM 프롬프트용 텍스트 생성
    # 3a. LLM에 전달할 '가장 최신 사례' 1건 찾기
    latest_case = None
    if final_cases_meta:
        try:
            # 'date' (YYYY-MM-DD 형식)를 기준으로 최신 날짜의 사례를 찾습니다.
            latest_case = max(
                (c for c in final_cases_meta if c.get('date', '0000-00-00') != 'N/A'), 
                key=lambda x: x.get('date', '0000-00-00'),
                default=None
            )
        except Exception:
            # 날짜 비교 중 오류 발생 시, 그냥 첫 번째 사례를 사용 (안전장치)
            latest_case = final_cases_meta[0]
        
    # 3b. LLM 프롬프트용 텍스트(retrieved_text) 생성
    if latest_case:
        # LLM에는 최신 사례 1건만 전달
        retrieved_text = f"[유사 시정 사례]"
        
        # --- [최종 수정: 모든 정보 제공하되 명확한 라벨링] ---
        case_summary_parts = []
        
        explanation = str(latest_case.get('explanation', '')).strip()
        conclusion = str(latest_case.get('conclusion', '')).strip()
        content = str(latest_case.get('content', '')).strip() # 약관 조항 원문
        
        # 1. 불공정 약관 조항 (문맥 파악용 - 나쁜 예시임을 명시)
        if content:
             case_summary_parts.append(f"  * [불공정 약관 조항 (참고용)]: {content}")
        
        # 2. 시정 요청 사유 (법적 논리)
        if explanation:
             case_summary_parts.append(f"  * [시정 요청 사유 (위법 사유)]: {explanation}")
                 
        # 3. 심사 결론 (수정 가이드)
        if conclusion:
             case_summary_parts.append(f"  * [심사 결론 (수정 방향)]: {conclusion}")
            
        # 리스트에 담긴 내용을 줄바꿈으로 합치기
        final_summary = "\n".join(case_summary_parts)
        
        retrieved_text += f"\n- 사례{latest_case['index']} (유사도 {latest_case['similarity']:.1%}):\n{final_summary}\n"
        
        if latest_case['related_law']:
            retrieved_text += f"  (관련법: {latest_case['related_law']})\n"
            
    else:
        # 검색된 사례가 없으면 0건으로 표시
        retrieved_text = f"[유사 시정 사례] (0건)\n"

    retrieved_text += f"\n[관련 법령] ({len(final_laws_meta)}건)\n"
    for l in final_laws_meta:
        retrieved_text += f"- 법령{l['index']} (유사도 {l['similarity']:.1%}): {l['content']}\n"
    
    print("\n" + "="*60)
    print(f"[DEBUG] AI에게 전달되는 참고 자료(retrieved_text):\n{retrieved_text}")
    print("="*60 + "\n")
    
    return {
        "related_cases": retrieved_text,
        "retrieved_cases_metadata": final_cases_meta,
        "retrieved_laws_metadata": final_laws_meta
    }

    
def generate_proposal_node(state: ContractState):
    print(f"[노드5] Solar API - 개선안 생성 (반복: {state['iteration']}/{MAX_ITERATIONS})\n")

    # 상태에서 필요한 정보 추출
    unfair_type = state['unfair_type']
    cases_meta = state.get('retrieved_cases_metadata', [])
    laws_meta = state.get('retrieved_laws_metadata', [])
    # original_clause = state['cleaned_text']
    
    
    # 최종 출력 문자열(final_output) 조립 시작 ---
    
    # 0. 불공정 여부 판단
    final_output = "### 0. 불공정 여부 판단\n"
    final_output += f"❌ **{unfair_type}**\n" # '불공정'이 확정적이므로 바로 ❌ 처리
    
    cases_str, laws_str = format_rag_results(
        cases_meta, 
        laws_meta, 
        state['fairness_label'] # "불공정"
    )
    final_output += cases_str
    final_output += laws_str

     # --- 2. "공정"할 경우, 여기서 완료 ---
    if unfair_type == "공정":
        print("공정 조항으로 판단되어 개선안 생성 없이 완료.\n")
        return {"improvement_proposal": final_output}

    # --- 3. "불공정"할 경우, LLM을 호출하여 개선안 + 표 생성 ---
    print("불공정 조항으로 판단되어 LLM 개선안 생성 시작...\n")
    
    # 사용자 피드백 (재시도 시)
    feedback_context = ""
    if state.get('modify_reason'):
        feedback_context = f"\n[추가 사용자 피드백]\n{state['modify_reason']}\n위 의견을 반영해 다시 작성하세요.\n"
        
    # LLM에 전달할 프롬프트 (인라인 diff 및 테이블 생성 요청)
    prompt = ACTIVE_GENERATE_PROPOSAL_PROMPT(
        original_clause=state['cleaned_text'],
        unfair_type=unfair_type,
        related_cases=state.get('related_cases', ""),
        feedback_context=feedback_context
    )
      # LLM 호출
    llm_response = llm.invoke(prompt).content
    
    # LLM 응답을 최종 출력에 추가
    final_output += f"\n{llm_response}"
    
    print("LLM 개선안 생성 완료.\n")
    
    return {"improvement_proposal": final_output}

def generate_fair_report_node(state: ContractState):
    """[노드 5.5] '공정' 판별 시, RAG 결과를 포함한 "공정" 리포트 생성"""
    print(f"\n[노드 5.5] 공정 판별 완료. RAG 결과를 포함한 리포트 생성.\n")
    
    # 0. 불공정 여부 판단
    final_output = "### 0. 불공정 여부 판단\n"
    final_output += "✅ **공정**\n"
    final_output += "\n입력하신 조항은 특별한 불공정 요소가 발견되지 않았습니다.\n"
    
    cases_meta = state.get('retrieved_cases_metadata', [])
    laws_meta = state.get('retrieved_laws_metadata', [])
    
    # --- 1. 유사한 사례 (대조 사례) ---  
    # --- 2. 쿼리와 유사한 법령 ---
    cases_str, laws_str = format_rag_results(
        cases_meta, 
        laws_meta, 
        state['fairness_label'] # "공정"
    )
    final_output += cases_str
    final_output += laws_str

    return {"improvement_proposal": final_output}
    
def interrupt_for_feedback_node(state: ContractState):
    """[노드6] HITL - Streamlit 피드백 대기"""
    print(f"\n[노드6] 피드백 대기 (Interrupt)\n")
    # Streamlit UI에서 피드백을 받을 때까지 그래프 실행을 일시 중지
    # return interrupt()
      # [수정]
    # Streamlit UI가 개선안 등 현재 state를 받아 렌더링할 수 있도록
    # interrupt()에 'state' 객체를 value로 전달합니다.

    return interrupt(value=state)

def process_feedback_node(state: ContractState):
    """[노드7] 사용자 피드백 처리 및 저장 (Streamlit에서 재개 시 실행)"""
    print(f"\n[노드7] 피드백 처리\n")
    
    feedback = state['user_feedback']
    retry_action = state.get('retry_action', '')
    current_iteration = state.get('iteration', 1)
    
    if feedback == "approved":
        save_result(state, "approved", current_iteration, total_iterations=current_iteration)
        print("[노드7] 결과 저장 완료 (수락)\n")
        return {"user_feedback": "approved", "retry_action": ""}
    
    elif feedback == "rejected":
        if retry_action == "retry":
            new_iteration = current_iteration + 1
            save_result(state, "rejected_retry", current_iteration)
            print(f"[노드7] 거절 기록 (재시도 예정) -> 반복 {new_iteration}차 진행\n")
            return {"user_feedback": "rejected", "iteration": new_iteration, "retry_action": "retry"}
        else:
            save_result(state, "rejected_discard", current_iteration, total_iterations=current_iteration)
            print(f"[노드7] 결과 저장 완료 (거절 및 폐기)\n")
            return {"user_feedback": "rejected", "retry_action": "discard"}
    
    elif feedback == "modify":
        if current_iteration >= MAX_ITERATIONS:
            save_result(state, "max_iteration_reached", current_iteration, total_iterations=current_iteration, modify_reason="반복 횟수 제한 도달")
            print(f"[노드7] 반복 횟수 제한 도달\n")
            return {"user_feedback": "approved", "retry_action": ""} # 강제 수락
        
        new_iteration = current_iteration + 1
        save_result(state, "modify_request", current_iteration, modify_reason=state.get('modify_reason', ''))
        print(f"[노드7] 수정 요청 저장 -> 반복 {new_iteration}차 진행\n")
        return {
            "user_feedback": "modify",
            "iteration": new_iteration,
            "modify_reason": state.get('modify_reason', ''),
            "retry_action": ""
        }
    
    return {"user_feedback": feedback, "retry_action": ""}