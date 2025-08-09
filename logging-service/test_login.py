import requests
from bs4 import BeautifulSoup

BASE_URL = "http://localhost:8080"


def get_csrf_and_cookies(session: requests.Session):
    r = session.get(f"{BASE_URL}/login", timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    token_el = soup.find("input", {"name": "csrf_token"})
    token = token_el["value"] if token_el else ""
    return token


def try_login(username: str, password: str):
    with requests.Session() as s:
        token = get_csrf_and_cookies(s)
        data = {"username": username, "password": password}
        if token:
            data["csrf_token"] = token
        r = s.post(f"{BASE_URL}/login", data=data, timeout=15, allow_redirects=False)
        return r.status_code, r.headers.get("Location"), r.text[:200]


if __name__ == "__main__":
    code, loc, body = try_login("admin", "Admin@123!")
    print({"status": code, "location": loc, "body": body})
