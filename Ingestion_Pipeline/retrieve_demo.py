"""Quick manual check: embed a query, search Qdrant, print what comes back.
Usage: python retrieve_demo.py "what are my chest exercises"
       python retrieve_demo.py --chunk-type weight_exercise,cardio "list my leg exercises"
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import store_qdrant

TOP_K = 5
BASE_FILTERS = {"client_id": "client_self", "active": True}


async def main(query, chunk_types=None):
    filters = dict(BASE_FILTERS)
    if chunk_types:
        filters["chunk_type"] = chunk_types
    points = await store_qdrant.search(query, top_k=TOP_K, filters=filters)
    print(f"query: {query!r} -- {len(points)} result(s)\n")
    for i, point in enumerate(points, 1):
        payload = point.payload
        print(f"[{i}] score={point.score:.3f} chunk_type={payload['chunk_type']} page={payload['page']} page_topic={payload['page_topic']}")
        print(f"    embed_text: {payload['embed_text']}")
        if payload.get("entity"):
            print(f"    entity: {payload['entity']}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    chunk_types = None
    if args and args[0] == "--chunk-type":
        chunk_types = args[1].split(",")
        args = args[2:]
    query = " ".join(args) or "what are my chest exercises"
    asyncio.run(main(query, chunk_types))
