from typing import Callable, Any
from functools import wraps
import time


def track_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0: float = time.time()
        result: Any = func(self, *args, **kwargs)
        self.duration = time.time() - t0
        return result
    return wrapper

