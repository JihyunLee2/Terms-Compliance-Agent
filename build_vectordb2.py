# pip install langchain_upstage langchain_chroma langgraph
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
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if any(
            pattern in line
            for pattern in [
                "법제처", "국가법령정보센터", "제1장", "제2장", "전문개정", "[시행", "[법률",
            ]
        ):
            continue
        if line.strip().isdigit():
            continue
        if line.strip():
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def parse_date_safe(date_val):
    """YYYY-MM-DD 형식 날짜를 안전하게 파싱"""
    try:
        if pd.isna(date_val):
            return None

        # pandas Timestamp인 경우 그대로 반환
        if isinstance(date_val, pd.Timestamp):
            return date_val

        date_str = str(date_val).strip()
        
        # utc=False 대신 명시적으로 처리
        parsed = pd.to_datetime(date_str, errors="coerce")
        
        # NaT 체크
        if pd.isna(parsed):
            return None
            
        return parsed

    except Exception as e:
        print(f"날짜 파싱 오류 - 입력값: {date_val}, 오류: {str(e)}")
        return None


def build_vectordb():
    print("벡터 DB 생성 중...")

    embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    documents = []

    # PDF 처리
    pdf_files = [
        "1_약관법.pdf",
        "1-2_약관심사지침.pdf",
        "2_금융소비자법시행령.pdf",
        "3_금융소비자보호에관한감독규정.pdf",
        "4_전자금융거래법.pdf",
    ]

    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            print(f"경고: {pdf_file}을(를) 찾을 수 없습니다")
            continue

        print(f"처리 중: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            cleaned_content = clean_page_content(chunk.page_content)
            if len(cleaned_content) < 30:
                continue

            chunk.page_content = cleaned_content
            chunk.metadata["source_type"] = "law"
            chunk.metadata["source_file"] = pdf_file
            documents.append(chunk)

    print(f"법령 처리 완료: {len(documents)}개 청크\n")

    # 엑셀 파일 처리
    print("불공정 사례 처리 중... (엑셀)")
    excel_path = "test251105.xlsx"
    target_sheets = ["불공정사례", "laws", "case", "불공정_공정 분류"]

    df = None  # 명시적 초기화
    
    if not os.path.exists(excel_path):
        print(f"경고: 엑셀 파일 '{excel_path}'이(가) 없어 사례 처리 skip.\n")
        df = pd.DataFrame()
    else:
        try:
            xls = pd.ExcelFile(excel_path)
            sheets_to_process = [s for s in target_sheets if s in xls.sheet_names]
            
            if not sheets_to_process:
                print(f"경고: 처리 대상 시트 {target_sheets}가 엑셀 파일에 없어 사례 처리 skip.\n")
                df = pd.DataFrame()
            else:
                print(f"처리 대상 시트: {sheets_to_process}")
                df_list = []
                for sheet in sheets_to_process:
                    try:
                        sheet_df = pd.read_excel(xls, sheet)
                        sheet_df['source_sheet'] = sheet  # source_sheet 컬럼 추가
                        df_list.append(sheet_df)
                    except Exception as e:
                        print(f"경고: 시트 '{sheet}' 읽기 실패 - {str(e)}")
                        continue
                
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    print(f"총 사례 수: {len(df)} (시트: {sheets_to_process})\n")
                else:
                    df = pd.DataFrame()
        except Exception as e:
            print(f"엑셀 파일 처리 오류: {str(e)}")
            df = pd.DataFrame()

    # 빈 DataFrame 처리
    if df is None or df.empty:
        print("처리할 사례 데이터가 없습니다.\n")
        valid_dates = []
        latest_date_str = None
        latest_timestamp = None
        successful_cases = 0
    else:
        print("=" * 60)
        print("[디버그] 샘플 날짜 확인")
        print("=" * 60)

        valid_dates = []
        if "보도시점" in df.columns:
            sample_dates = df["보도시점"].dropna().head(5).tolist()
            print(f"샘플 보도시점 5개: {sample_dates}")
            
            for i, date_val in enumerate(sample_dates):
                parsed = parse_date_safe(date_val)
                print(f"샘플 {i+1}: 입력={date_val}, 파싱됨={parsed}")

            failed_count = 0
            for date_val in df["보도시점"].dropna():
                parsed = parse_date_safe(date_val)
                if parsed is not None:
                    valid_dates.append(parsed)
                else:
                    failed_count += 1

            print(f"파싱 결과: 성공 {len(valid_dates)}개, 실패 {failed_count}개\n")
        else:
            print("보도시점 컬럼 없음. 날짜 검증은 건너뜁니다.\n")

        # 최신 날짜 계산 (중복 호출 제거)
        if valid_dates:
            latest_date = max(valid_dates)
            latest_date_str = latest_date.strftime("%Y-%m-%d")
            latest_timestamp = int(latest_date.timestamp())
            print(f"✓ 최신 보도시점: {latest_date_str}")
            print(f"✓ Unix Timestamp: {latest_timestamp}\n")
        else:
            print(f"경고: 유효한 날짜가 없습니다.\n")
            latest_date_str = None
            latest_timestamp = None

        # 사례 처리
        successful_cases = 0
        for idx, row in df.iterrows():
            try:
                term_text = str(row.get("약관 조항", "")).strip()
                conclusion_text = str(row.get("시정 요청 결론", "")).strip()
                detail_text = str(row.get("시정 요청 설명", "")).strip()

                # 모든 필드가 비어있으면 스킵
                if not term_text and not conclusion_text and not detail_text:
                    continue

                # 날짜 처리
                report_date = None
                if "보도시점" in df.columns:
                    report_date = parse_date_safe(row.get("보도시점"))

                date_str = report_date.strftime("%Y-%m-%d") if report_date else None
                timestamp = int(report_date.timestamp()) if report_date else None

                page_content = (
                    f"약관: {term_text}\n\n"
                    f"결론: {conclusion_text}\n\n"
                    f"상세내용: {detail_text}"
                )

                # 안전한 값 추출
                explanation = row.get("시정 요청 설명", "")
                case_type = row.get("유형", "")
                related_law = row.get("관련법(약관법)", "")
                ref_law = row.get("참고 법", "")
                ref_explanation = row.get("참고 법 설명", "")
                source_sheet = row.get("source_sheet", "")

                metadata = {
                    "source_type": "case",
                    "source_sheet": source_sheet if not pd.isna(source_sheet) else "",
                    "case_id": idx,
                    "date": date_str,
                    "date_timestamp": timestamp,
                    "case_type": case_type if not pd.isna(case_type) else "",
                    "explanation": explanation if not pd.isna(explanation) else "",
                    "conclusion": conclusion_text,
                    "related_law": related_law if not pd.isna(related_law) else "",
                    "reference_law": ref_law if not pd.isna(ref_law) else "",
                    "detailed_info": (
                        ref_explanation if not pd.isna(ref_explanation) else ""
                    ),
                    "regulatory_period": "2024-11-26 이후 적용 예정",
                }

                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
                successful_cases += 1

            except Exception as e:
                print(f"경고: {idx}번 사례 처리 실패 - {str(e)}")
                continue

        print(f"사례 처리 완료: 총 {successful_cases}개\n")

    # 벡터 DB 저장
    print("벡터 DB 저장 중...")
    
    if not documents:
        print("경고: 저장할 문서가 없습니다!")
        return None
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="contract_laws",
    )

    # 설정 파일 저장
    config_data = {
        "latest_case_date": latest_date_str,
        "latest_case_timestamp": latest_timestamp,
        "db_update_time": datetime.now().isoformat(),
        "total_cases": successful_cases,
        "total_documents": len(documents),
    }

    with open("vectordb_config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"벡터 DB 생성 완료!")
    print(f"설정 파일 저장 완료\n")
    print(f"최신 사례 기준: {latest_date_str}")
    print(f"Unix Timestamp: {latest_timestamp}")
    print(f"총 사례 수: {successful_cases}개")
    print(f"총 문서 수: {len(documents)}개\n")

    return vectorstore


if __name__ == "__main__":
    try:
        result = build_vectordb()
        if result is None:
            print("벡터 DB 생성 실패")
            exit(1)
    except Exception as e:
        print(f"치명적 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)