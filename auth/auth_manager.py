import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import os

# 설정 파일 경로 (절대 경로로 안전하게 지정)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def get_authenticator():
    """
    config.yaml 파일을 읽어 인증 객체(authenticator)를 생성해 반환합니다.
    """
    # 1. Config 파일 로드
    try:
        with open(CONFIG_PATH) as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error(f"⚠️ 설정 파일({CONFIG_PATH})을 찾을 수 없습니다.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ 설정 파일 로드 중 오류 발생: {e}")
        st.stop()

    # 2. 디버깅용 출력 (필요하다면 함수 안에서 실행해야 함)
    # print(f"로드된 사용자: {list(config['credentials']['usernames'].keys())}")

    # 3. 인증 객체 생성
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    
    return authenticator

def check_login_status(authenticator):
    """
    현재 세션의 로그인 상태를 확인하고 처리합니다.
    Return: True(로그인 성공), False(실패/미로그인)
    """
    # 1. 로그인 위젯 표시
    try:
        authenticator.login(fields={'username': '사내 이메일', 'password': '비밀번호'})
    except Exception as e:
        st.error(e)
        return False

    # 2. 상태 확인
    if st.session_state["authentication_status"]:
        # [로그인 성공]
        with st.sidebar:
            st.success(f"환영합니다, **{st.session_state['name']}**님")
            authenticator.logout('로그아웃', 'sidebar')
            st.divider()
        return True
    
    elif st.session_state["authentication_status"] is False:
        # [로그인 실패]
        st.error('❌ 사내 이메일 또는 비밀번호가 일치하지 않습니다.')
        return False
        
    elif st.session_state["authentication_status"] is None:
        # [로그인 전]
        st.warning('🔒 관계자 외 접속을 금지합니다. 사내 이메일로 로그인하세요.')
        return False
    
    return False