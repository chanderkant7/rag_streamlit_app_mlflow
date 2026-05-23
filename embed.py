import asyncio

import httpx

from auth import get_access_token
from clients import create_embeddings_client


async def main():
    async with httpx.AsyncClient() as client:
        access_token = await get_access_token(client)

    embed_client = create_embeddings_client(access_token)
    embeddings = embed_client.embed_query("Hello world!")
    print(embeddings)


asyncio.run(main())
