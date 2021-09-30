import requests

url = "https://aip.baidubce.com/oauth/2.0/token"

API_Key = "lajGYERl5lC3d7f2ZS0FC1kh"
Secret_Key = "GT8ZEotQ0n2uAPUrxiRkZZ7IMUcAdCIL"

data = {
    'grant_type':'client_credentials',
    'client_id':API_Key,
    'client_secret':Secret_Key,
}

response = requests.post(url=url,data=data)  # 获取百度API的token
print(response.text)