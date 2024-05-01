import asyncio
import os
from generate_content import call_openai_gpt4

async def test_call_openai_gpt4():
    prompt = "Dime cómo integrar un script de python con un GPT de OpenAI"
    response = await call_openai_gpt4(prompt)
    print(response)

if __name__ == "__main__":
    asyncio.run(test_call_openai_gpt4())
