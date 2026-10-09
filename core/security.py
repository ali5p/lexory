import hashlib
import os

from fastapi import Request
from redis import Redis

redis_client = Redis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=6379, 
    decode_responses=True
)

def get_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip

    x_forwarded = request.headers.get("x-forwarded-for")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"

def get_device_fingerprint_prod(request: Request) -> str:

    ip_address = get_client_ip(request)
    
    user_agent = request.headers.get("user-agent", "unknoown-ua").strip().lower()
    accept_lang = request.headers.get("access-language", "unknown-lang").strip().lower()
    accept_encoding = request.headers.get("accept-encoding", "unknown-enc").strip().lower()

    raw_fingerprint = f"{user_agent}|{accept_lang}|{accept_encoding}|{ip_address}"

    return hashlib.sha256(raw_fingerprint.encode("utf-8")).hexdigest()
