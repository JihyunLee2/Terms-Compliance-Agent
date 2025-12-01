import streamlit as st
from auth import auth_manager
from langgraph_components import load_app_safe

from ui_modules.pdf_module import run_pdf_batch_mode
from ui_modules.chat_ui import run_chatbot_mode
from ui_modules.guide_ui import draw_user_guide, draw_analysis_scope
from config2 import SIMILARITY_THRESHOLD
        
@st.cache_resource
def get_cached_app():
    """
    LangGraph 앱과 VectorStore를 캐싱하여
    Streamlit이 Rerun 되어도 메모리(Checkpoint)가 유지되도록 합니다.
    """
    return load_app_safe()

def main_chatbot_ui():
    st.set_page_config(page_title="신용카드 약관 분석 AI", layout="wide", initial_sidebar_state="collapsed")
    
    # --- [1] 상태 초기화 ---
    if "show_guide" not in st.session_state:
        st.session_state.show_guide = False
    if "show_scope" not in st.session_state:
        st.session_state.show_scope = False

    # 현재 '도움말 모드'인지 확인 (가이드나 범위 화면 중 하나라도 켜져 있으면 True)
    is_help_mode = st.session_state.show_guide or st.session_state.show_scope
    
    # ---------------------------------------------------------
    # [사이드바 영역]
    # ---------------------------------------------------------
    with st.sidebar:
       # 1. 검색 설정 (가이드나 범위 화면이 아닐 때만 활성화)
        disabled_status = st.session_state.show_guide or st.session_state.show_scope
        
        st.subheader("검색 옵션")
        similarity_threshold_percent = st.slider(
            "유사도 임계값 (%)",
            min_value=0,
            max_value=100,
            value=int(SIMILARITY_THRESHOLD * 100),
            step=5,
            format="%d%%",
            disabled=is_help_mode
        )
        current_threshold_value = similarity_threshold_percent / 100.0
        
        if not is_help_mode:
            st.caption(f"현재 설정: {similarity_threshold_percent}% 이상 유사 사례 검색")
        
        st.divider()
            
        st.header("도움말")
        
        # 2. 화면 전환 버튼 로직 (가이드 보기 / 분석 범위 / 돌아가기)
        # 2-1. 가이드 버튼 (보고 있으면 '닫기', 안 보고 있으면 '열기')
        if st.session_state.show_guide:
            # 현재 가이드를 보고 있는 상태 -> '돌아가기' 버튼으로 표시
            if st.button("**⬅️ 돌아가기**", use_container_width=True):
                st.session_state.show_guide = False
                st.rerun()
        else:
            # 가이드를 안 보고 있는 상태 -> '가이드 보기' 버튼으로 표시
            if st.button("사용 가이드 보기", use_container_width=True):
                st.session_state.show_guide = True
                st.session_state.show_scope = False # 다른 창은 닫음
                st.rerun()

        # 2-2. 분석 범위 버튼 (보고 있으면 '닫기', 안 보고 있으면 '열기')
        if st.session_state.show_scope:
            # 현재 분석 범위를 보고 있는 상태 -> '돌아가기' 버튼으로 표시
            if st.button("**⬅️ 돌아가기**", use_container_width=True, key="btn_close_scope"):
                st.session_state.show_scope = False
                st.rerun()
        else:
            # 분석 범위를 안 보고 있는 상태 -> '범위 보기' 버튼으로 표시
            if st.button("데이터 구조 / 판단 기준 보기", use_container_width=True):
                st.session_state.show_scope = True
                st.session_state.show_guide = False # 다른 창은 닫음
                st.rerun()
    
        st.caption("2025.11 약관 분석 모듈 v1.0")

    # ---------------------------------------------------------
    # [메인 화면 영역]
    # ---------------------------------------------------------
    
    # [A] 가이드 보기 모드일 때 -> 가이드 함수 호출
    if st.session_state.show_guide:
        draw_user_guide()
    
    # [B] 분석 범위 보기 모드
    elif st.session_state.show_scope:
        draw_analysis_scope()
    
    # [C] 분석 모드일 때 -> 기존 탭(Radio) 화면 표시
    else:
        st.title("신용카드 약관 분석 AI")
        st.caption("본 서비스는 법무팀의 신규 약관 작성을 지원하는 내부용 도구입니다. AI 분석은 법적 해석을 대체하지 않으며, 최종 검토·판단 책임은 법무팀 담당자에게 있습니다.")

        
        # 앱 로드
        app, vectorstore = load_app_safe()
        if not app or not vectorstore:
            st.error("앱 초기화 실패")
            return

        # --- 기존의 Radio 탭 유지 ---
        tab_options = ["💬 챗봇 (단일 조항 분석)", "📄 PDF (전체 문서 분석)"]
        
        # 탭 상태 유지
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = tab_options[0]

        active_tab = st.radio(
            "모드 선택",
            tab_options,
            key="active_tab", # session_state와 자동 연동
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.divider()

        if active_tab == "💬 챗봇 (단일 조항 분석)":
            run_chatbot_mode(app, current_threshold_value)
            
        elif active_tab == "📄 PDF (전체 문서 분석)":
            run_pdf_batch_mode(app, vectorstore, current_threshold_value)
        

def main():
    # 1. 인증 관리자로부터 객체 가져오기
    authenticator = auth_manager.get_authenticator()

    # 2. 로그인 상태 확인 및 처리 (이 함수가 로그인 창 표시부터 검증까지 다 함)
    if auth_manager.check_login_status(authenticator):
        # 3. 로그인 성공 시 메인 UI 실행
        main_chatbot_ui()

if __name__ == "__main__":
    main()