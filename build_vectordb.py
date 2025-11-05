# 시정요청상세내용도 벡터디비에 넣기, 청크사이즈 250으로 줄이기

# build_vectordb.py

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json

load_dotenv()

def clean_page_content(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if any(pattern in line for pattern in [
            '법제처', '국가법령정보센터', '제1장', '제2장',
            '전문개정', '[시행', '[법률'
        ]):
            continue
        if line.strip().isdigit():
            continue
        if line.strip():
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

def parse_date_safe(date_val):
    """YYYY-MM-DD 형식 날짜를 안전하게 파싱"""
    try:
        if pd.isna(date_val):
            return None
        
        date_str = str(date_val).strip()
        
        parsed = pd.to_datetime(date_str, format='%Y-%m-%d')
        return parsed
    
    except Exception as e:
        print(f"⚠ 날짜 파싱 오류 - 입력값: {date_val}, 오류: {str(e)}")
        return None

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
        if not os.path.exists(pdf_file):
            print(f"경고: {pdf_file}을(를) 찾을 수 없습니다")
            continue
        
        print(f"처리 중: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        
        for chunk in chunks:
            cleaned_content = clean_page_content(chunk.page_content)
            if len(cleaned_content) < 30:
                continue
            
            chunk.page_content = cleaned_content
            chunk.metadata['source_type'] = 'law'
            chunk.metadata['source_file'] = pdf_file
            documents.append(chunk)
    
    print(f"법령 처리 완료: {len(documents)}개 청크\n")
    
    print("불공정 사례 처리 중...")
    df = pd.read_csv('1028ver2.csv', encoding='utf-8')
    print(f"총 사례 수: {len(df)}\n")
    
    print("=" * 60)
    print("[디버그] 샘플 날짜 확인")
    print("=" * 60)
    sample_dates = df['보도시점'].dropna().head(5).tolist()
    for i, date_val in enumerate(sample_dates):
        parsed = parse_date_safe(date_val)
        print(f"샘플 {i+1}: 입력={date_val}, 파싱됨={parsed}")
    print("=" * 60 + "\n")
    
    valid_dates = []
    failed_count = 0
    
    for date_val in df['보도시점'].dropna():
        parsed = parse_date_safe(date_val)
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
            
            report_date = parse_date_safe(row['보도시점'])
            if report_date is None:
                continue
            
            date_str = report_date.strftime('%Y-%m-%d')
            timestamp = int(report_date.timestamp())
            
            page_content = f"약관: {term_text}\n\n결론: {conclusion_text}"
            
            explanation = row.get('시정 요청 설명', '')
            case_type = row.get('유형', '')
            related_law = row.get('관련법(약관법)', '')
            ref_law = row.get('참고 법', '')
            ref_explanation = row.get('참고 법 설명', '')
            
            metadata = {
                'source_type': 'case',
                'case_id': idx,
                'date': date_str,
                'date_timestamp': timestamp,
                'case_type': case_type if not pd.isna(case_type) else '',
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
        persist_directory="./chroma_db",
        collection_name="contract_laws"
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
