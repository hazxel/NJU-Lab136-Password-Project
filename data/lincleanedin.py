import json

text_file_path = './linkedin_with_hash.txt'
json_file_path = './linkedin_cleaned.json'
passwords = []
with open(text_file_path, 'r',encoding='utf-8') as file:
    while True:
        line = file.readline()
        if line:
            passwords.append(line.strip().split(':')[1])
        else:
            break

with open(json_file_path, 'w',encoding='utf-8') as json_file:
    json.dump(passwords, json_file, ensure_ascii = True)
