# Prepare the data payload as a JSON string
  data_payload = '{"accessKey":"msvvucw2qall55ftt3u80xru8qiz8506","request":{"dob":' + dob + ',"bank_id":' + bank_id + ',"city":' + city + ',"total_transaction":' + total_transaction + ',"income":' + income'}}'

import json

# Prepare the data payload as a JSON string
data_payload = json.dumps({
    "accessKey": "msvvucw2qall55ftt3u80xru8qiz8506",
    "request": {
        "dob": dob,
        "bank_id": bank_id,
        "city": city,
        "total_transaction": total_transaction,
        "income": income
    }
})