import time
from typing import Dict, Tuple
from collections import defaultdict


class TokenBucketRateLimiter:
    """
    Sliding window token bucket rate limiter.
    Limits requests per IP / client identifier over a rolling time window.
    """

    def __init__(self, max_requests: int = 15, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_ip -> list of timestamps
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Tuple[bool, int, float]:
        """
        Checks whether client_id can make a request.
        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Prune timestamps older than window
        timestamps = [ts for ts in self.requests[client_id] if ts > window_start]
        self.requests[client_id] = timestamps

        if len(timestamps) < self.max_requests:
            self.requests[client_id].append(now)
            remaining = self.max_requests - len(self.requests[client_id])
            return True, remaining, 0.0

        # Over limit
        oldest_request = timestamps[0]
        retry_after = round(self.window_seconds - (now - oldest_request), 1)
        return False, 0, max(retry_after, 1.0)
