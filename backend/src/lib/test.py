import requests

url = "https://genai.rcac.purdue.edu/api/chat/completions"
headers = {
    "Authorization": f"Bearer {"sk-dc9144edcd5446bdb2092a08faccadc8"}",
    "Content-Type": "application/json"
}
body = {
    "model": "llama3.1:latest",
    "messages": [
        {
            "role": "user",
            "content": "What is your name?"
        }
    ],
    "stream": False
}
response = requests.post(url, headers=headers, json=body)
if response.status_code == 200:
    print(response.text)
else:
    raise Exception(f"Error: {response.status_code}, {response.text}")
