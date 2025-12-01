import bcrypt

# 1. 암호화할 비밀번호
passwords = ['1234']

print("\n" + "="*40)
print("🔑 비밀번호 해시 생성기")
print("="*40)

for password in passwords:
    # bcrypt를 사용하여 비밀번호 해싱
    # (streamlit-authenticator와 동일한 방식입니다)
    hashed_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    hashed_str = hashed_bytes.decode('utf-8')
    
    print(f"원본: {password}")
    print(f"해시: {hashed_str}")
    print("-" * 40)
    print("▼ 아래 줄을 복사해서 config.yaml의 password 항목에 넣으세요 ▼")
    print(hashed_str)
    print("=" * 40 + "\n")