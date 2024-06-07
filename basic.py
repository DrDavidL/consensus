import streamlit as st
from openai import OpenAI
import asyncio
import aiohttp
import json
from embedchain import App

from prompts import system_prompt_expert_questions, expert1_system_prompt, expert2_system_prompt, expert3_system_prompt

# Set your OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]

# async def get_response(query):
#     async with aiohttp.ClientSession() as session:
#         response = await session.post(
#             'https://api.openai.com/v1/chat/completions',
#             headers={
#                 'Authorization': f'Bearer {api_key}',
#                 'Content-Type': 'application/json',
#             },
#             json={
#                 'model': 'gpt-3.5-turbo',
#                 'messages': [{'role': 'user', 'content': query}],
#             }
#         )
#         return await response.json()

async def get_response(messages):
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'gpt-4o',
                'messages': messages,
                'temperature': 0.3,
            }
        )
        return await response.json()

async def get_responses(queries):
    tasks = [get_response(query) for query in queries]
    return await asyncio.gather(*tasks)


def extract_expert_info(json_input):
    # Parse the JSON input
    data = json.loads(json_input)
    
    # Initialize empty lists to hold the extracted information
    experts = []
    domains = []
    expert_questions = []
    
    # Iterate over the rephrased questions and extract the information
    for item in data['rephrased_questions']:
        experts.append(item['expert'])
        domains.append(item['domain'])
        expert_questions.append(item['question'])
    
    return experts, domains, expert_questions

def create_chat_completion(
    messages,
    model="gpt-4o",
    frequency_penalty=0,
    logit_bias=None,
    logprobs=False,
    top_logprobs=None,
    max_tokens=None,
    n=1,
    presence_penalty=0,
    response_format=None,
    seed=None,
    stop=None,
    stream=False,
    include_usage=False,
    temperature=1,
    top_p=1,
    tools=None,
    tool_choice="none",
    user=None
):
    client = OpenAI()

    # Prepare the parameters for the API call
    params = {
        "model": model,
        "messages": messages,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "max_tokens": max_tokens,
        "n": n,
        "presence_penalty": presence_penalty,
        "response_format": response_format,
        "seed": seed,
        "stop": stop,
        "stream": stream,
        "temperature": temperature,
        "top_p": top_p,
        "user": user
    }

    # Handle the include_usage option for streaming
    if stream:
        params["stream_options"] = {"include_usage": include_usage}
    else:
        params.pop("stream_options", None)

    # Handle tools and tool_choice properly
    if tools:
        params["tools"] = [{"type": "function", "function": tool} for tool in tools]
        params["tool_choice"] = tool_choice

    # Handle response_format properly
    if response_format == "json_object":
        params["response_format"] = {"type": "json_object"}
    elif response_format == "text":
        params["response_format"] = {"type": "text"}
    else:
        params.pop("response_format", None)

    # Remove keys with None values
    params = {k: v for k, v in params.items() if v is not None}
    
    completion = client.chat.completions.create(**params)
    
    return completion

def check_password() -> bool:
    def password_entered() -> None:
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            app = App()
            app.reset()
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    st.session_state.setdefault("password_correct", False)
    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key='password')
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False

    return True

def main():
    st.title('Simultaneous LLM Queries')
    
    if check_password():
    
        # Obtain the initial query from the user
        original_query = st.text_input('Original Query', 'How long should alendronate be held for tooth extractions?')
        find_experts_messages = [{'role': 'system', 'content': system_prompt_expert_questions}, 
                                {'role': 'user', 'content': original_query}]
        if st.button('Begin Research'):
            completion = create_chat_completion(messages=find_experts_messages, temperature=0.3, response_format="json_object")
            with st.sidebar:
                with st.expander("Experts Identified"):
                    st.write(f"**Response:**")
                    json_output = completion.choices[0].message.content
                    st.write(json_output)
                    experts, domains, expert_questions = extract_expert_info(json_output)
                    st.write(f"**Experts:** {experts}")
                    st.write(f"**Domains:** {domains}")
                    st.write(f"**Expert Questions:** {expert_questions}")
            
            updated_expert1_system_prompt = expert1_system_prompt.format(expert=experts[0], domain=domains[0])
            updated_expert2_system_prompt = expert2_system_prompt.format(expert=experts[1], domain=domains[1])
            updated_expert3_system_prompt = expert3_system_prompt.format(expert=experts[2], domain=domains[2])
            updated_question1 = expert_questions[0]
            updated_question2 = expert_questions[1]
            updated_question3 = expert_questions[2]
            
            expert1_messages = [{'role': 'system', 'content': updated_expert1_system_prompt}, 
                                {'role': 'user', 'content': updated_question1}]
            expert2_messages = [{'role': 'system', 'content': updated_expert2_system_prompt}, 
                                {'role': 'user', 'content': updated_question2}]
            expert3_messages = [{'role': 'system', 'content': updated_expert3_system_prompt}, 
                                {'role': 'user', 'content': updated_question3}]
            
            with st.spinner('Waiting for experts to respond...'):
                expert_answers = asyncio.run(get_responses([expert1_messages, expert2_messages, expert3_messages]))
            for i, response in enumerate(expert_answers):
                with st.expander(f"AI {experts[i]} Perspective"):
                    st.write(response['choices'][0]['message']['content'])
                # st.write(f"**Expert {i+1} Response:**")
                # st.write(response['choices'][0]['message']['content'])

        # query1 = st.text_input('Query 1', 'What is the capital of France?')
        # query2 = st.text_input('Query 2', 'Explain the theory of relativity.')
        # query3 = st.text_input('Query 3', 'What are the benefits of a ketogenic diet?')

        # if st.button('Send Queries'):
        #     queries = [query1, query2, query3]
        #     responses = asyncio.run(get_responses(queries))

        #     for i, response in enumerate(responses):
        #         st.write(f"**Response {i+1}:**")
        #         st.write(response['choices'][0]['message']['content'])

if __name__ == '__main__':
    main()
