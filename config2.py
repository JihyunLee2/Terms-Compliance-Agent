# config.py
"""설정 파일
환경에 상관없이 거의 변하지 않는 상수를 포함합니다.
"""

import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- LLM 및 임베딩 모델 ---
if "UPSTAGE_API_KEY" not in os.environ:
    raise EnvironmentError("환경변수에 UPSTAGE_API_KEY가 설정되지 않았습니다.")

EMBEDDING_MODEL = "solar-embedding-1-large-passage"
LLM_MODEL = "solar-pro2"

llm = ChatUpstage(model=LLM_MODEL)
embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)

# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")


# --- 상수 ---
MAX_ITERATIONS = 4 # HITL(수정 요청) 최대 반복 횟수 # 4 (초안 1회 + 수정 기회 3회)
FAIRNESS_MAX_ITERATIONS = 3   # 공정/불공정 분류 재시도 최대 횟수
CONFIDENCE_THRESHOLD = 0.8    # 공정/불공정 분류 반복 중단 기준 신뢰도


# --- 벡터DB 검색 설정 (RAG 관련) ---
SIMILARITY_THRESHOLD = 0.4  # 유사도 임계점 (0.0 ~ 1.0), Streamlit 기본 유사도 임계값 (UI에서 변경 가능)
SEARCH_TOP_K_CASES = 10     # 초기 검색 개수 (필터링 전)
SEARCH_TOP_K_LAWS = 10
MAX_DISPLAY_CASES = 5       # 최종 표시 개수
MAX_DISPLAY_LAWS = 5
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "contract_laws"
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


# --- Streamlit UI 설정 ---
SHOW_SIMILARITY_SCORES = True   # 유사도 점수 표시 여부
SHOW_RETRIEVED_CASES = True     # 검색된 사례 표시 여부