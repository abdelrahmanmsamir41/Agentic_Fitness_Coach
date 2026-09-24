"""Lazy-singleton helper -- the same "construct an expensive client/model
once, on first use" shape was hand-written independently in several modules
(store_qdrant.py, extraction/extract.py, exercise_reference/embeddings.py).
"""


def lazy(factory):
    """Returns a zero-arg getter that calls factory() once and caches the result."""
    cache = []

    def get():
        if not cache:
            cache.append(factory())
        return cache[0]

    return get


def _demo():
    calls = []
    get = lazy(lambda: calls.append(1) or object())
    a = get()
    b = get()
    assert a is b, "must return the same cached instance"
    assert len(calls) == 1, "factory must run exactly once"


if __name__ == "__main__":
    _demo()
    print("ok")
