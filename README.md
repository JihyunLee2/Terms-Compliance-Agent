# 4th-Project
최종

# 신용카드 약관 검토 AI 에이전트 (Project Title)

이 프로젝트는 LangGraph와 Streamlit을 기반으로 한 AI 에이전트로, 신용카드 약관의 불공정성을 검토하고 개선안을 제안합니다.

## 1. 주요 기능

* **실시간 조항 검토:** 사용자가 입력한 약관 조항 1개를 즉시 분석하여 불공정 여부, 유형, 개선안을 제공합니다.
* **PDF 일괄 검토:** PDF 약관 파일 전체를 업로드하면, 내부 조항들을 자동으로 분리하여 불공정 의심 조항 목록을 리포트합니다.
    * (참고: 메인 애플리케이션(app.py)과 PDF 일괄 검토 모듈(pdf_module.py)은 **아민** 님이 작성한 초기 코드를 기반으로 모듈화했으며, 기존 탭(Tab) UI를 라디오 버튼 방식으로 개선했습니다.)

## 5. 환경 변수 (`.env`) 설정

이 프로젝트는 Upstage API와 LangSmith 모니터링을 위한 API 키가 필요합니다. 프로젝트 루트(최상위 폴더)에 `.env` 파일을 생성하고 아래 내용을 각자의 키 값으로 채워주세요.

⚠️ **중요:** `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다. **절대 API 키를 코드나 Git에 직접 노출하지 마세요.**

**`.env` 파일 예시:**

```dotenv
# .env

# 1. Upstage API 키 (Embeddings)
UPSTAGE_API_KEY=your_upstage_api_key_here

# 2. LangSmith 설정 (디버깅 및 모니터링용)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=project_name

## 폴더 구조
/
├── .env                    # 환경 변수 파일 (로컬 전용)
├── app.py                  # 메인 Streamlit 앱
├── config2.py              # 설정 (API 키, DB 경로)
├── utils.py                # 공통 유틸 함수
├── requirements.txt        # 의존성 목록
├── README.md               # 프로젝트 소개
├── .gitignore              # Git 제외 파일
│
├── langgraph_components/   # LangGraph logic
│   ├── __init__.py
│   ├── graph.py            # 그래프 정의
│   ├── nodes.py            # 노드 로직
│   ├── state.py            # 상태 정의
│   └── prompts.py          # LLM 프롬프트
│
├── ui_modules/             # Streamlit UI 하위 모듈
│   ├── __init__.py
│   └── pdf_module.py       # PDF 일괄처리 UI
│ 
├── scripts/                # 일회성 실행 스크립트
│   └── build_vectordb.py   # Vector DB 생성
│ 
├── data/                   # <-- (신규) DB 생성용 원본 데이터
│   ├── 1_약관법.pdf
│   ├── 1-2_약관심사지침.pdf
│   ├── 2_금융소비자법시행령.pdf
│   ├── 3_금융소비자보호에관한감독규정.pdf
│   ├── 4_전자금융거래법.pdf
│   └── kftc_unfair_terms_cases.csv # 불공정 사례 데이터셋
│
└── (chroma_db/)