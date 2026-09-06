import requests

def test_query_service_alpha():
    # upload file for query
    with open("data/pdfs/tenant_alpha/fictional_company_contract.pdf", "rb") as f:
        requests.post(
            "http://127.0.0.1:8000/upload/", 
            files={"file": f}, 
            data={"tenant_id": "tenant_alpha", "tenant_token": "tenant_alpha_token"})
    
    with open("data/pdfs/tenant_beta/fictional_company_performance_report.pdf", "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/upload/", 
            files={"file": f}, 
            data={"tenant_id": "tenant_beta", "tenant_token": "tenant_beta_token"})

    response = requests.post(
        "http://127.0.0.1:8000/question/", 
        json={
            "question": "What is the company name?", 
            "tenant_id": "tenant_alpha", 
            "tenant_token": "tenant_alpha_token"
            }
        )
    response_json = response.json()
    assert response.status_code == 200
    assert response_json["question"] == "What is the company name?"
    assert "Northstar Dynamics" in response_json["answer"]

    response = requests.post(
        "http://127.0.0.1:8000/question/", 
        json={
            "question": "What is company name?", 
            "tenant_id": "tenant_beta", 
            "tenant_token": "tenant_beta_token"
            }
        )
    response_json = response.json()

    assert response.status_code == 200
    assert response_json["question"] == "What is company name?"
    assert "Asteron" in response_json["answer"]

