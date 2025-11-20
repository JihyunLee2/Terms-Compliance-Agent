from __future__ import annotations
import re
import os
import json
import pandas as pd
from datetime import datetime
import pypdf # 1. pdf_module에서 이동
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple # 2. 타입 힌트 추가
from langchain_chroma import Chroma
from config2 import embeddings, CHROMA_DB_PATH, COLLECTION_NAME, CHROMA_COLLECTION_METADATA


# --- 상수 (nodes.py에서 이동) ---

LAW_FILENAME_MAP = {
    "1_약관법.pdf": "약관법",
    "1-2_약관심사지침.pdf": "약관심사지침",
    "2_금융소비자법시행령.pdf": "금융소비자법 시행령",
    "3_금융소비자보호에관한감독규정.pdf": "금융소비자보호 감독규정",
    "4_전자금융거래법.pdf": "전자금융거래법" 
}


# --- 헬퍼 함수 (nodes.py, build_vectordb.py, pdf_module.py 등에서 이동) ---

def is_valid_contract_clause(clause: str) -> tuple[bool, str]:
    """Rule-based 입력 검증기  (nodes.py에서 이동)"""
    clause = clause.strip()
    if len(clause) < 20:
        return False, "입력이 너무 짧습니다 (최소 20자 필요)"
    
    contract_keywords = [
        '조항', '조건', '약관', '규정', '제', '항', '조', '자', '금지', '가능', '불가', 
        '의무', '책임', '권리', '계약', '해지', '중단', '변경', '환불', '배상', '배제', 
        '면책', '수수료', '이용료', '결제', '할인', '서비스', '제공', '개인정보', '보호', 
        '이용', '관리', '통지', '동의', '유효', '기간', '상효', '시행', '효력', '청구', 
        '위반', '손해배상', '면책조항', '이용자', '회사', '당사자', '할부거래', '회원', 
        '고객', '연회비', '신용카드'
    ]  # 11/15  '할부거래', '회원', '고객', '연회비', '신용카드' 추가
    
    if not any(keyword in clause for keyword in contract_keywords):
        return False, "약관 관련 키워드 미검출 (예: 조항, 약관, 조건, 의무 등)"
    
    if any(q in clause for q in ['?', '？']): # 전각, 반각
        return False, "질문 형식으로 보입니다. 약관 조항을 입력해주세요."
    
    if any(q in clause for q in ['안녕하세요', '반갑습니다', '뭐해', '뭐야',
        '어때', '날씨', '오늘', '내일', '궁금해', '알려줘',
        '재밌', '슬프', '기쁘', '화나']):
        return False, "일상 대화 형식으로 보입니다. 약관 조항을 입력해주세요."
    
    return True, "검증 통과"

def save_result(state: dict, status: str, iteration: int,
                modify_reason: str = "", total_iterations: int = None):
    """
    [수정됨] 결과를 logs 폴더에 저장합니다. (에러 방지 및 디버깅 강화)
    """
    try:
        # 1. 데이터 추출 (KeyError 방지 위해 .get 사용)
        result = {
            "timestamp": datetime.now().isoformat(),
            "session_id": state.get('session_id', 'unknown'),
            "user_email": state.get('user_email', 'unknown'),
            "user_name": state.get('user_name', 'unknown'),
            "status": status,
            "iteration": iteration,
            "total_iterations": total_iterations or iteration,
            # --- 중요: state에 해당 키가 없으면 빈 문자열 처리 ---
            "original_clause": state.get('clause', ''),
            "cleaned_text": state.get('cleaned_text', ''),
            "unfair_type": state.get('unfair_type', ''),
            "improvement_proposal": state.get('improvement_proposal', ''),
            "user_feedback": state.get('user_feedback', ''),
            "modify_reason": modify_reason or state.get('modify_reason', '')
        }

        # 2. 폴더 생성
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 3. 파일 쓰기 (feedback_log.jsonl 하나에 누적)
        filename = os.path.join(log_dir, "feedback_log.jsonl")
        
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
        print(f"✅ [Log Saved] {status} 저장 성공 -> {filename}")

    except Exception as e:
        # 에러가 나도 멈추지 않고 콘솔에만 출력
        print(f"❌ [Log Error] 로그 저장 실패: {str(e)}")

def extract_text_from_pdf(uploaded_file):
    """PDF 파일에서 텍스트를 추출합니다.  (pdf_module.py에서 이동)"""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n\n"
        return pdf_text
    except Exception as e:
        print(f"PDF 텍스트 추출 실패: {e}")# st.error 대신 print 사용
        return ""    

def split_text_into_clauses(full_text: str) -> List[str]:
    """
    전체 텍스트를 '제N조' 단위로 분할 (Regex 활용)
    """
    # 1. 제N조 패턴으로 분할 (Lookahead assertion (?=...) 사용으로 '제N조' 텍스트 보존)
    # 패턴 설명: 줄바꿈(\n) 뒤에 공백(\s*)이 있을 수도 있고 '제' + 숫자 + '조'가 나오는 구간
    article_pattern = re.compile(r'(?=\n\s*제\s*\d+\s*조)')
    
    # 전체 텍스트에서 맨 앞의 불필요한 공백 제거
    full_text = full_text.strip()
    
    # 분할 실행
    chunks = article_pattern.split(full_text)
    
    valid_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk: continue
        
        # 2. 너무 긴 조항(1000자 이상)은 2차 분할 (RecursiveCharacterTextSplitter)
        if len(chunk) > 1000:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            valid_chunks.extend(sub_splitter.split_text(chunk))
        else:
            # 3. 너무 짧은 쓰레기 텍스트(20자 미만) 버리기
            if len(chunk) >= 20:
                valid_chunks.append(chunk)
                
    return valid_chunks

def clean_page_content(text: str) -> str:
    """
    PDF 페이지별 노이즈(헤더, 푸터, 페이지 번호) 제거
    """
    # 1. 줄 단위로 분리
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line: continue # 빈 줄 제거
        
        # 2. 법령 PDF 특유의 노이즈 패턴 제거 (법제처, 날짜, 페이지 번호 등)
        # 예: "법제처", "국가법령정보센터", "1", "2024. 2. 6." 등 숫자만 있는 줄
        if any(keyword in line for keyword in [
            '법제처', '국가법령정보센터', '제1장', '제2장'
            , '[시행', '[법률', '전문개정', '타법개정', '일부개정'
            ]):
            continue
            
        # 3. 숫자만 있거나(페이지번호), 특수문자로 감싸진 페이지 번호 제거 (- 1 -)
        if re.match(r'^[\-\s]*\d+[\-\s]*$', line):
            continue
            
        cleaned_lines.append(line)
        
    return '\n'.join(cleaned_lines)

def parse_date_safe(date_val):
    """YYYY-MM-DD 형식 날짜를 안전하게 파싱  build_vectordb.py에서 이동"""
    try:
        if pd.isna(date_val):
            return None
        
        date_str = str(date_val).strip()
        
        parsed = pd.to_datetime(date_str, format='%Y-%m-%d')
        return parsed
    
    except Exception as e:
        print(f"⚠ 날짜 파싱 오류 - 입력값: {date_val}, 오류: {str(e)}")
        return None


def format_rag_results(
    cases_meta: List[Dict[str, Any]], 
    laws_meta: List[Dict[str, Any]], 
    fairness_label: str
) -> Tuple[str, str]:
    """
    RAG 검색 결과(사례, 법령)를 리포트용 Markdown 문자열로 포맷팅합니다.
    '공정'/'불공정' 여부에 따라 제목과 접두사를 다르게 처리합니다.
    
    Args:
        cases_meta: state['retrieved_cases_metadata']
        laws_meta: state['retrieved_laws_metadata']
        fairness_label: state['fairness_label'] ("공정" 또는 "불공정")

    Returns:
        (cases_output, laws_output) 튜플
    """
    
    # --- 1. 유사 사례 포맷팅 ---
    # '공정'/'불공정'에 따라 제목과 접두사를 우선 설정
    if fairness_label == "공정":
        cases_output = "\n### 1. 참고 사례 (동일 주제)\n"
        # cases_output += "동일 주제로 검색된 '불공정' 시정 사례입니다. 입력 조항은 아래 사례와 달리 공정한 것으로 보입니다.\n"
        case_prefix = "[불공정 사례] "
    else:
        cases_output = "\n### 1. 유사한 사례\n"
        case_prefix = ""

    # '공정'/'불공정'에 따라 제목과 접두사를 우선 설정
    if not cases_meta:
        # 1-1. 사례가 없을 때
        if fairness_label == "공정":
            cases_output += "입력 조항과 동일한 주제의 '불공정' 시정 사례를 찾지 못했습니다.\n"
        else:
            cases_output += "관련 사례를 찾지 못했습니다.\n"
    else:
        # 1-2. 사례가 있을 때
        # [수정] '공정'일 때만 서두 문장을 '사례가 있을 때' 추가
        if fairness_label == "공정":
            cases_output += "동일 주제로 검색된 '불공정' 시정 사례입니다. 입력 조항은 아래 사례와 달리 공정한 것으로 보입니다.\n"
        
        # 사례 목록 출력
        for case in cases_meta:
            # 원본 로직 (nodes.py)
            case_summary = case['content'].split('약관:')[1].split('결론:')[0].strip()
            if len(case_summary) > 70:
                case_summary = case_summary[:70] + "..."
            
            cases_output += f"* **`(유사도 {case['similarity']:.0%})`** {case_prefix}{case_summary}\n"

    # --- 2. 참고 법령 포맷팅 (공정/불공정 공통) ---
    laws_output = "\n### 2. 참고 법령\n"
    if not laws_meta:
        laws_output += "관련 법령을 찾지 못했습니다.\n"
    else:
        for law in laws_meta:
            # 원본 로직 (nodes.py)
            similarity = law['similarity']            
            content = law['content'].strip()
            metadata = law.get('metadata', {})
            source_file = metadata.get('source_file', '알 수 없는 법령')
            
            # utils.py 상단의 LAW_FILENAME_MAP 상수 사용
            law_name = LAW_FILENAME_MAP.get(source_file, source_file)
            
            if len(content) > 70:
                content = content[:70] + "..."
            
            laws_output += f"* **`(유사도 {similarity:.0%})`** **`[{law_name}]`** - {content}\n"
            
    return cases_output, laws_output

def clean_clause_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r'^[\s•\-\*]+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def parse_fairness_response(response: str) -> tuple[str, float]:
    lines = response.strip().split('\n')
    
    fairness_label = "불공정"
    logit = 0.0
    
    if len(lines) >= 1:
        fairness_label = lines[0].strip()
    
    if len(lines) >= 2:
        try:
            logit = float(lines[1].strip())
        except ValueError:
            logit = 0.0
    
    return fairness_label, logit