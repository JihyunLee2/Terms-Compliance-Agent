import streamlit as st
import traceback
from datetime import datetime
from langchain_core.tracers.context import tracing_v2_enabled

# 모듈화된 설정, 그래프, PDF 모듈 로드
from config2 import SIMILARITY_THRESHOLD, MAX_ITERATIONS, SHOW_RETRIEVED_CASES
from langgraph_components import load_app_safe
from ui_modules import run_pdf_batch_mode

def run_chatbot_mode(app, current_threshold_value):
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
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": """### 안녕하세요, 법률 약관 검토 챗봇입니다👋\n
새로운 약관 조항의 공정성 검토를 도와드리겠습니다. 분석을 원하는 **약관 조항**만 아래 채팅창에 입력해 주세요.
            
        [입력 예시]
        회원이 본 카드의 발급 목적과 다르게 이용한다고 카드사가 판단하거나, 
        기타 이에 준하는 중대한 사유가 발생하여 계약 유지가 곤란하다고 인정되는 경우, 카드사는 본 계약을 해지할 수 있습니다.
        """
        })

    # 1. 채팅 메시지 기록을 먼저 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 2. RAG 결과(유사 사례)가 state에 존재할 경우, expander를 출력
    # (RAG 실행 전에는 'cases'가 None이므로 이 블록은 건너뜀)
    cases = st.session_state.current_state.get('retrieved_cases_metadata', None)
    
    if SHOW_RETRIEVED_CASES and cases is not None:
        used_threshold = st.session_state.current_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
        
        with st.expander("참고한 유사 사례 보기", expanded=False):
            if cases:
                st.caption(f"총 {len(cases)}개 사례 (유사도 {used_threshold:.0%} 이상)")
                
                for case in cases:
                    similarity = case['similarity']
                    
                    if similarity >= 0.7:
                        color = "🟢"
                    elif similarity >= 0.5:
                        color = "🟡"
                    else:
                        color = "🟠"
                    
                    st.markdown(f"### {color} 사례 {case['index']} - 유사도: {similarity:.1%}")
                    st.caption(f"📅 {case['date']} | 유형: {case['case_type']}")
                    
                    with st.container():
                        st.markdown("**불공정 약관 조항:**")
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
    
    # 3. 피드백 대기 상태(hitl_pending)인 경우, 피드백 UI 출력
    if st.session_state.hitl_pending:
        current_iteration = st.session_state.current_state.get('iteration', 1)
        
        # --- [UI 상태 관리 변수 초기화] ---
        if "show_modify_input" not in st.session_state:
            st.session_state.show_modify_input = False

        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")

        # ============================================================
        # [화면 A] 기본 버튼 선택 화면 (입력창 숨김 상태)
        # ============================================================
        if not st.session_state.show_modify_input:
            col1, col2, col3 = st.columns(3)
            
            # 1. 수락 버튼
            with col1:
                if st.button("현재 개선안 수락 (Approve)", use_container_width=True, type="primary"):
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

            # 2. 수정/재생성 버튼 (누르면 입력창 열림)
            with col2:
                if st.button("다른 개선안 생성 (Modify)", use_container_width=True):
                    st.session_state.show_modify_input = True  # 상태 변경
                    st.rerun()

            # 3. 폐기 버튼
            with col3:
                if st.button("현재 개선안 폐기 (Discard)", use_container_width=True):
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

        # ============================================================
        # [화면 B] 수정 사유 입력 화면 (버튼 누른 후)
        # ============================================================
        else:
            st.markdown("### 📝 수정 요청 사항 입력")
            st.caption("구체적으로 적어주실수록 더 정확한 개선안이 나옵니다.")
            
            # 반복 횟수 제한 체크
            if current_iteration >= MAX_ITERATIONS:
                st.warning(f"⚠️ 반복 횟수 제한({MAX_ITERATIONS}회)에 도달하여 더 이상 수정할 수 없습니다.")
                if st.button("돌아가기", use_container_width=True):
                    st.session_state.show_modify_input = False
                    st.rerun()
            else:
                modify_reason = st.text_area(
                    "수정 요청 사유:", 
                    key="modify_reason_input",
                    height=150,
                    placeholder="예) 위약금 비율을 조금 더 낮춰주세요.\n예) 해지 사유를 더 구체적으로 명시해주세요."
                )

                b_col1, b_col2 = st.columns([1, 1])
                
                with b_col1:
                    if st.button("취소 (이전으로)", use_container_width=True):
                        st.session_state.show_modify_input = False
                        st.rerun()
                        
                with b_col2:
                    if st.button("제출하기", type="primary", use_container_width=True):
                        if not modify_reason.strip():
                            st.error("수정 요청 사유를 입력해주세요.")
                        else:
                            # 제출 로직
                            st.session_state.pending_feedback = {
                                "user_feedback": "modify",
                                "modify_reason": modify_reason.strip(),
                                "retry_action": ""
                            }
                            st.session_state.hitl_pending = False
                            st.session_state.show_modify_input = False # 상태 초기화
                            
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": f"[피드백] 수정 요청:\n{modify_reason.strip()}"
                            })
                            st.rerun()
                            
        st.chat_input("피드백을 먼저 완료해주세요.", disabled=True)

    # 4. 피드백 대기 상태가 아닌 경우, 채팅 입력창 활성화
    else:
        # 4-1. 보류 중인 피드백이 있다면 먼저 처리
        if st.session_state.pending_feedback is not None:
            feedback_input = st.session_state.pending_feedback
            st.session_state.pending_feedback = None
            
            # 피드백 입력(invoke) 시, 현재 사이드바의 임계값을 다시 주입(overwrite)합니다.
            feedback_input["similarity_threshold"] = current_threshold_value  # 10/16 추가
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.chat_message("assistant"):
                with st.spinner("피드백을 반영하여 처리 중..."):
                    try:
                        output = app.invoke(feedback_input, config=config)
                        st.session_state.current_state = output
                        
                        last_feedback = output.get('user_feedback', '')
                        last_retry = output.get('retry_action', '')

                        if last_feedback == "approved" or (last_feedback == "rejected" and last_retry == "discard"):
                            st.markdown("### 검토 완료\n검토가 최종 완료되었습니다.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "검토가 완료되었습니다."
                            })
                            st.rerun()
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

        # 4-2. 새 프롬프트(쿼리)를 받음
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
                        
                        if output.get('validation_failed', False):
                            error_msg = f"입력 오류: {output.get('cleaned_text', '알 수 없는 오류')}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.thread_id = None
                        # --- 수정 11/15---
                        # '공정'일 때와 '불공정'일 때를 분리
                        elif output.get('fairness_label') == "공정":
                            st.session_state.current_state = output
                            # '공정'일 경우 (generate_fair_report_node 경유)
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            # '공정'이므로 피드백 대기(HITL) 없이 완료
                            st.session_state.hitl_pending = False 
                            st.session_state.thread_id = None # 세션 종료
                            st.rerun()
                        else:
                            st.session_state.current_state = output
                            # '불공정'일 경우 (generate_proposal_node 경유)
                            st.markdown("### 제안 (첫 번째 개선안)")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            # '불공정'이므로 피드백 대기(HITL)
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"약관 검토 중 오류 발생: {e}")
                        st.exception(traceback.format_exc())
                        st.session_state.thread_id = None
                        st.session_state.hitl_pending = False

def main_chatbot_ui():
    st.set_page_config(page_title="약관 검토 챗봇", layout="wide")
    st.title("약관 검토 챗봇")
    st.caption("본 서비스는 법무팀의 신규 약관 작성을 지원하는 내부용 도구입니다. AI 분석은 법적 해석을 대체하지 않으며, 최종 검토·판단 책임은 법무팀 담당자에게 있습니다.")
    
    with st.sidebar:
        st.header("검색 설정")
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
        st.subheader("정보")
        st.markdown(
            """
            * **모델:** Solar-Pro2
            * **버전:** 약관 분석 모듈 v1.0
            * **최근 업데이트:** 2025.11
            * **성능 범위:** 불공정 여부 판단, 유사 사례/법령 검색, 개선안 생성
            """
        )
    
    # 모듈화된 load_app_safe 호출
    app, vectorstore = load_app_safe()
    if not app or not vectorstore:
        st.error("애플리케이션을 초기화하지 못했습니다. 설정을 확인하세요.")
        return

    # --- 1. (수정) st.tabs 대신 st.radio로 탭 상태 관리 ---
    # st.radio는 'key'를 지원하므로 페이지 Rerun 시에도 상태가 유지됩니다.
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "💬 챗봇 (단일 조항 검토)" # 기본값 설정

    tab_options = ["💬 챗봇 (단일 조항 검토)", "📄 PDF (전체 문서 검토)"]
    
    # horizontal=True와 label_visibility="collapsed"로 탭처럼 보이게 함
    active_tab = st.radio(
        "모드 선택",
        tab_options,
        key="active_tab", # session_state와 연결
        horizontal=True,
        label_visibility="collapsed" # '모드 선택' 레이블 숨기기
    )
    
    st.divider() # 탭과 내용 구분

    # --- 2. (수정) 'with tab1/tab2:' 대신 if/elif 구문 사용 ---   
    if active_tab == "💬 챗봇 (단일 조항 검토)":
        run_chatbot_mode(app, current_threshold_value)
        
    elif active_tab == "📄 PDF (전체 문서 검토)":
        # 모듈화된 pdf_module 호출
        run_pdf_batch_mode(app, vectorstore, current_threshold_value)
        

if __name__ == "__main__":
    main_chatbot_ui()