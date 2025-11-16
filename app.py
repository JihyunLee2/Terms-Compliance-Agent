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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.hitl_pending:
        current_iteration = st.session_state.current_state.get('iteration', 1)
        used_threshold = st.session_state.current_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
        
        if SHOW_RETRIEVED_CASES:
            with st.expander("참고한 유사 사례 보기", expanded=False):
                cases = st.session_state.current_state.get('retrieved_cases_metadata', [])
                
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
        
        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")
        
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
                        # --- 수정 11/15---
                        # '공정'일 때와 '불공정'일 때를 분리
                        elif output.get('fairness_label') == "공정":
                            # '공정'일 경우 (generate_fair_report_node 경유)
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            # '공정'이므로 피드백 대기(HITL) 없이 완료
                            st.session_state.hitl_pending = False 
                            st.session_state.thread_id = None # 세션 종료
                        else:
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
                        st.session_state.current_state = {}


def main_chatbot_ui():
    st.set_page_config(page_title="약관 검토 챗봇", layout="wide")
    st.title("약관 검토 챗봇")
    
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