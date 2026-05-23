import time

import httpx

from config import AUTH_URL, SCOPE, GRANT_TYPE, CLIENT_ID, CLIENT_SECRET

# Module-level cache: (access_token, expiry_timestamp)
_token_cache: tuple[str, float] | None = None
_EXPIRY_BUFFER = 60  # refresh this many seconds before actual expiry


async def get_access_token(client: httpx.AsyncClient) -> str:
    global _token_cache

    if _token_cache is not None:
        token, expiry = _token_cache
        if time.monotonic() < expiry:
            return token

    body = {
        "grant_type": GRANT_TYPE,
        "scope": SCOPE,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = await client.post(AUTH_URL, headers=headers, data=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    expires_in = data.get("expires_in", 3600)
    _token_cache = (token, time.monotonic() + expires_in - _EXPIRY_BUFFER)
    return token
