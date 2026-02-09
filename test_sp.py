# Test SharePoint Connection
import requests
import msal
import os
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from app.config import (
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET,
    SHAREPOINT_SITE_ID, SHAREPOINT_LIST_ID
)

print(f"Tenant: {AZURE_TENANT_ID}")
print(f"Client: {AZURE_CLIENT_ID}")
print(f"Site ID: {SHAREPOINT_SITE_ID}")
print(f"List ID: {SHAREPOINT_LIST_ID}")

# 1. Get Token
authority = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
app = msal.ConfidentialClientApplication(
    AZURE_CLIENT_ID, authority=authority, client_credential=AZURE_CLIENT_SECRET
)
result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

if "access_token" not in result:
    print("❌ Failed to get token")
    print(result.get("error"))
    print(result.get("error_description"))
    exit()

print("✅ Token acquired")
token = result["access_token"]

# 2. Test Site Access
headers = {"Authorization": f"Bearer {token}"}
print("\nTesting Site Access...")
site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE_ID}"
resp = requests.get(site_url, headers=headers)
print(f"Site Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.json().get('error', {}).get('message')}")
else:
    print("✅ Site accessible")

# 3. Test List Access
print("\nTesting List Access...")
list_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE_ID}/lists/{SHAREPOINT_LIST_ID}"
resp = requests.get(list_url, headers=headers)
print(f"List Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.json().get('error', {}).get('message')}")
else:
    print("✅ List accessible")
