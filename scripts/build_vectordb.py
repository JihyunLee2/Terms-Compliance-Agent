import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# 이 스크립트(build_vectordb.py)가 있는 폴더의 부모 폴더(프로젝트 루트)를
# 파이썬 경로에 추가합니다.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json
import utils

load_dotenv()

# --- [추가/수정] 데이터 폴더 경로 지정 ---
# 이 스크립트(build_vectordb.py)가 있는 'scripts' 폴더의
# 부모 폴더(루트)를 기준으로 'data' 폴더 경로를 설정합니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


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

# ------------------------------------

def build_vectordb():
    print("벡터 DB 생성 중...")
    
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    documents = []
    
    pdf_files = [
        "1_약관법.pdf",
        "1-2_약관심사지침.pdf",
        "2_금융소비자법시행령.pdf",
        "3_금융소비자보호에관한감독규정.pdf",
        "4_전자금융거래법.pdf"
    ]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        if not os.path.exists(file_path):
            print(f"경고: {pdf_file}을(를) 찾을 수 없습니다")
            continue
        
        print(f"처리 중: {pdf_file}")
        loader = PyPDFLoader(file_path)
        pages = loader.load() # 페이지 단위 로드
        
        # --- 페이지 병합 (Merge) ---
        full_text = ""
        for page in pages:
            # 1. 페이지별 헤더/푸터 노이즈 제거
            cleaned_content = utils.clean_page_content(page.page_content)
            # 2. 전체 텍스트로 합치기 (줄바꿈 추가)
            full_text += f"\n{cleaned_content}"
            
        # --- [핵심 변경 사항] 조항 단위 분할 (Split) ---
        chunks = utils.split_text_into_clauses(full_text)
        
        # 분할된 텍스트를 Document 객체로 변환
        for chunk_text in chunks:
            # 메타데이터 생성 (어떤 법령인지)
            metadata = {
                'source_type': 'law',
                'source_file': pdf_file,
                'law_priority': assign_law_priority(pdf_file) # 우선순위 함수를 호출하여 메타데이터에 저장
            }
    
            # Document 객체 생성
            doc = Document(page_content=chunk_text, metadata=metadata)
            documents.append(doc)
            
    print(f"법령 처리 완료: {len(documents)}개 청크\n")
    
    print("불공정 사례 처리 중...")
    # --- [수정] data 폴더에서 CSV 파일 읽기 ---
    csv_path = os.path.join(DATA_DIR, 'kftc_unfair_terms_cases.csv')
    df = pd.read_csv(csv_path, encoding='utf-8') # (수정)
    # df = pd.read_csv('kftc_unfair_terms_cases.csv', encoding='utf-8')  # 불공정 사례 115개 11/13
    print(f"총 사례 수: {len(df)}\n")
    
    print("=" * 60)
    print("[디버그] 샘플 날짜 확인")
    print("=" * 60)
    sample_dates = df['보도시점'].dropna().head(5).tolist()
    for i, date_val in enumerate(sample_dates):
        parsed = utils.parse_date_safe(date_val)
        print(f"샘플 {i+1}: 입력={date_val}, 파싱됨={parsed}")
    print("=" * 60 + "\n")
    
    valid_dates = []
    failed_count = 0
    
    for date_val in df['보도시점'].dropna():
        parsed = utils.parse_date_safe(date_val)
        if parsed is not None:
            valid_dates.append(parsed)
        else:
            failed_count += 1
    
    print(f"파싱 결과: 성공 {len(valid_dates)}개, 실패 {failed_count}개\n")
    
    if valid_dates:
        latest_date = max(valid_dates)
        latest_date_str = latest_date.strftime('%Y-%m-%d')
        latest_timestamp = int(latest_date.timestamp())
        print(f"✓ 최신 보도시점: {latest_date_str}")
        print(f"✓ Unix Timestamp: {latest_timestamp}\n")
    else:
        print(f"⚠ 경고: 유효한 날짜가 없습니다.\n")
        return None
    
    successful_cases = 0
    for idx, row in df.iterrows():
        try:
            term_text = row['약관 조항']
            conclusion_text = row['시정 요청 결론']
            
            if pd.isna(term_text) or pd.isna(conclusion_text):
                continue
            
            report_date = utils.parse_date_safe(row['보도시점'])
            if report_date is None:
                continue
            
            date_str = report_date.strftime('%Y-%m-%d')
            timestamp = int(report_date.timestamp())
            
            page_content = f"약관: {term_text}\n\n결론: {conclusion_text}"
            
            explanation = row.get('시정 요청 설명', '')
            case_type = row.get('대분류', '')  # 유형에서 현재 컬럼에 맞게 대분류로 변경   11/13
            related_law = row.get('관련법(약관법)', '')
            ref_law = row.get('참고 법', '')
            ref_explanation = row.get('참고 법 설명', '')
            fairness = row.get('공정여부', '')  # 공정여부 추가 11/13
            
            metadata = {
                'source_type': 'case',
                'case_id': idx,
                'date': date_str,
                'date_timestamp': timestamp,
                'case_type': case_type if not pd.isna(case_type) else '',
                'fairness': fairness if not pd.isna(fairness) else "",  # 공정여부 추가 11/13
                'explanation': explanation if not pd.isna(explanation) else '',
                'conclusion': conclusion_text,
                'related_law': related_law if not pd.isna(related_law) else '',
                'reference_law': ref_law if not pd.isna(ref_law) else '',
                'detailed_info': ref_explanation if not pd.isna(ref_explanation) else '',
                'regulatory_period': '2024-11-26 이후 적용 예정'
            }
            
            doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            
            documents.append(doc)
            successful_cases += 1
        
        except Exception as e:
            print(f"경고: {idx}번 사례 처리 실패 - {str(e)}")
            continue
    
    print(f"사례 처리 완료: 총 {successful_cases}개\n")
    
    print("벡터 DB 저장 중...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=os.path.join(BASE_DIR, "chroma_db"),
        collection_name="contract_laws",
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    config_data = {
        "latest_case_date": latest_date_str,
        "latest_case_timestamp": latest_timestamp,
        "db_update_time": datetime.now().isoformat(),
        "total_cases": successful_cases
    }
    
    with open('vectordb_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"벡터 DB 생성 완료!")
    print(f"설정 파일 저장 완료\n")
    print(f"최신 사례 기준: {latest_date_str}")
    print(f"Unix Timestamp: {latest_timestamp}")
    print(f"총 사례 수: {successful_cases}개\n")
    
    return vectorstore

if __name__ == "__main__":
    build_vectordb()
