import asyncio

import httpx

from auth import get_access_token
from clients import create_chat_client
from config import CHAT_MODEL


async def main():
    async with httpx.AsyncClient() as client:
        access_token = await get_access_token(client)

    oai_client = create_chat_client(access_token)
    messages = [{"role": "user", "content": "Hi, what is Prime number"}]
    response = oai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    print(response.model_dump_json(indent=2))


asyncio.run(main())