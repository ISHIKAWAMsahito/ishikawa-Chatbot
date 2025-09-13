import secrets

# 安全なランダム文字列（秘密鍵）を生成
# 32バイト = 256ビットで、十分な強度です
secret_key = secrets.token_urlsafe(32)

print("生成されたJWT秘密鍵 (これを.envにコピー):")
print(secret_key)