import requests
r = requests.post('https://modelservice.cml.apps.lintas.cloudeka.ai/model', data='{"accessKey":"msvvucw2qall55ftt3u80xru8qiz8506","request":{"dob":"19 - 22","bank_id":"022","city":"Jakarta","total_transaction":731064773,"income":7652685}}', headers={'Content-Type': 'application/json', 'Authorization': 'Bearer 3482e3829a3d03e244f6afcc92a1cb10d35882a99b4818650056c657a6549e72.cec1c3badecda98adfec248bb1b99767ec8206c28d6acdb85fdf0e4a4877f407'})

print(r)