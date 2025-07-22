import bcrypt

# Replace with your real raw passwords
passwords = {
    "akash": "test123",
    "john": "johnpass",
    "emma": "emma456"
}

for username, raw_pw in passwords.items():
    hashed_pw = bcrypt.hashpw(raw_pw.encode(), bcrypt.gensalt())
    print(f"{username}: {hashed_pw.decode()}")