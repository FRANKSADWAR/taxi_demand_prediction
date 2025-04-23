import requests

def get_receipts_data():
    headers = {
    "Authorization": "token 831bc07f0be234b:a57fe9b774d7051", 
    "Content-Type": "application/json"
    }
    url = "https://prd.erp.agribora.com/api/method/frappe.client.get_list"

    data = {
    "doctype": "WRS Receipt",
    "fields": ["serial_number", "depositor","commodity","refrence_batch", "quantity","grade","posting_date","status","unique_depositor_reference"]
    }
    response = requests.post(url, headers=headers, json=data)

    return response.json()

if __name__ == "__main__":
    wrsc_data = get_receipts_data()
    print(wrsc_data)