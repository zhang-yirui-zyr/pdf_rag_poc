import requests
import pytest

@pytest.mark.parametrize("tenant_id, tenant_token, file_path, expected", [
    ("tenant_alpha", "tenant_alpha_token", "data/pdfs/tenant_alpha/fictional_company_contract.pdf", 200),
    ("tenant_beta", "tenant_beta_token", "data/pdfs/tenant_beta/fictional_company_performance_report.pdf", 200),
    ("tenant_random", "tenant_random_token", "data/pdfs/tenant_beta/fictional_company_performance_report.pdf", 401),
])
def test_upload_service_success(tenant_id, tenant_token, file_path, expected):
    with open(file_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/upload/", 
            files={"file": f}, 
            data={"tenant_id": tenant_id, "tenant_token": tenant_token})
    assert response.status_code == expected


@pytest.mark.parametrize("tenant_id, tenant_token, file_path", [
    ("tenant_alpha", "tenant_alpha_token", "data/pdfs/tenant_alpha/fictional_company_contract.pdf"),
    ("tenant_beta", "tenant_beta_token", "data/pdfs/tenant_beta/fictional_company_performance_report.pdf"),
])
def test_upload_service_file_exists(tenant_id, tenant_token, file_path):
    with open(file_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/upload/", 
            files={"file": f}, 
            data={"tenant_id": tenant_id, "tenant_token": tenant_token})
        response_json = response.json()
    assert response_json["file"] == file_path.split("/")[-1]


@pytest.mark.parametrize("tenant_id, tenant_token, file_path", [
    ("tenant_alpha", "tenant_alpha_token", "data/pdfs/tenant_alpha/fictional_company_contract.pdf"),
    ("tenant_beta", "tenant_beta_token", "data/pdfs/tenant_beta/fictional_company_performance_report.pdf"),
])
def test_upload_service_records_exists(tenant_id, tenant_token, file_path):
    with open(file_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/upload/", 
            files={"file": f}, 
            data={"tenant_id": tenant_id, "tenant_token": tenant_token})
        response_json = response.json()
    assert response_json["saved_records"] >= response_json["num_chunks"]