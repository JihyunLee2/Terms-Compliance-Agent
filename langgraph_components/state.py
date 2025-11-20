from typing import TypedDict, List, Tuple, Dict, Any, Optional

class ContractState(TypedDict):
    """
    LangGraph의 상태를 정의하는 TypedDict입니다.
    """
    # 필수 입력
    clause: str                     # 원본 입력 조항
    session_id: str                 # LangSmith 추적을 위한 세션 ID
    iteration: int                  # 현재 반복 횟수 (피드백)
    
    # 노드 출력
    cleaned_text: str               # Rule-based 정제된 텍스트
    unfair_type: str                # 불공정 유형 (LLM 분류)
    related_cases: str              # RAG 검색 결과 (사례 + 법령 텍스트)
    improvement_proposal: str       # LLM이 생성한 개선안
    
    # RAG 메타데이터 (Streamlit UI용)
    retrieved_cases_metadata: List[Dict[str, Any]]
    retrieved_laws_metadata: List[Dict[str, Any]]
    similarity_threshold: float     # RAG 검색 시 사용할 유사도 임계값
    
    # 사용자 정보 필드
    user_email: Optional[str]
    user_name: Optional[str]
    
    # 공정/불공정 분류 관련
    fairness_label: str             # 최종 공정/불공정 분류 ("공정", "불공정")
    fairness_retry_count: int       # 공정/불공정 분류 반복 횟수
    fairness_confidence: float      # 공정/불공정 확신도
    results_history: List[Tuple[str, float]] # (classification: str, confidence: float) 튜플 리스트
    
    # HITL (Human-in-the-Loop)
    user_feedback: str              # 사용자 피드백 ("approved", "rejected", "modify")
    modify_reason: str              # "modify" 선택 시 수정 요청 사유
    retry_action: str               # "rejected" 선택 시 하위 액션 ("retry", "discard")
    
    # 상태 플래그
    validation_failed: bool         # 룰 기반 입력 검증 실패 여부