import hashlib
# ここにハッシュ化したいパスワードを入力
password = "ishikawa3150"
hashed_password = hashlib.sha256(password.encode()).hexdigest()
print(f"設定したいパスワード: {password}")
print(f"生成されたハッシュ値: {hashed_password}")