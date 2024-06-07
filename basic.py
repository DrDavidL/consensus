import streamlit as st
import openai
import asyncio
import aiohttp

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

async def get_response(query):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {openai.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': query}],
            }
        )
        return await response.json()

async def get_responses(queries):
    tasks = [get_response(query) for query in queries]
    return await asyncio.gather(*tasks)

def main():
    st.title('Simultaneous LLM Queries')

    query1 = st.text_input('Query 1', 'What is the capital of France?')
    query2 = st.text_input('Query 2', 'Explain the theory of relativity.')
    query3 = st.text_input('Query 3', 'What are the benefits of a ketogenic diet?')

    if st.button('Send Queries'):
        queries = [query1, query2, query3]
        responses = asyncio.run(get_responses(queries))

        for i, response in enumerate(responses):
            st.write(f"**Response {i+1}:**")
            st.write(response['choices'][0]['message']['content'])

if __name__ == '__main__':
    main()
