import redis
import json
import hashlib
# import os

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)

def _make_key(namespace: str, data: dict):
    hash_input = json.dumps(data, sort_keys=True)
    key_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    return f"{namespace}:{key_hash}"

# Cache assessment
def cache_assessment(request: dict, result: dict, ttl=3600):
    key = _make_key("assessment", request)
    redis_client.setex(key, ttl, json.dumps(result))

# Get cached assessment
def get_cached_assessment(request: dict):
    key = _make_key("assessment", request)
    result = redis_client.get(key)
    return json.loads(result) if result else None

# Cache context chunks
def cache_chunks(topic: str, chunks: list[str], ttl=3600):
    key = f"chunks:{topic.lower()}"
    redis_client.setex(key, ttl, json.dumps(chunks))

def get_cached_chunks(topic: str):
    key = f"chunks:{topic.lower()}"
    result = redis_client.get(key)
    return json.loads(result) if result else None

def get_user_difficulty(user_id: str) -> str:
    key = f"user_perf:{user_id}"
    data = redis_client.hgetall(key)
    
    correct = int(data.get(b"correct", 0))
    total = int(data.get(b"total", 0))
    
    if total == 0:
        return "medium"  # default

    accuracy = correct / total
    if accuracy > 0.8:
        return "hard"
    elif accuracy < 0.5:
        return "easy"
    else:
        return "medium"