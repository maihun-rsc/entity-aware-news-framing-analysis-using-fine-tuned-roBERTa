from curl_cffi import requests

url = "https://www.espncricinfo.com/story/sl-vs-ind-meet-saransh-jain-india-s-33-year-old-test-recruit-1547824"
try:
    response = requests.get(url, impersonate="chrome120")
    print("Status Code:", response.status_code)
    print("Length:", len(response.text))
except Exception as e:
    print("Error:", e)
