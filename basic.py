import asyncio
import json
import queue
import re
import tempfile
import threading

import aiohttp
import requests
import streamlit as st
from openai import OpenAI

from aiocache import Cache
from aiocache import cached

from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import (
    StreamingStdOutCallbackHandlerYield,
    generate,
)

from prompts import (
    system_prompt_expert_questions,
    expert1_system_prompt,
    expert2_system_prompt,
    expert3_system_prompt,
)

# Set your OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]



@st.cache_data
def realtime_search(query, domains, max):
    url = "https://real-time-web-search.p.rapidapi.com/search"
    full_query = f"{domains} {query}"
    querystring = {"q": full_query, "limit": max}
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com",
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        response_data = response.json().get('data', [])
        urls = [item.get('url') for item in response_data]
        snippets = [
            f"**{item.get('title')}**  \n*{item.get('snippet')}*  \n{item.get('url')} <END OF SITE>"
            for item in response_data
        ]
    except requests.exceptions.RequestException as e:
        st.error(f"RapidAPI real-time search failed to respond: {e}")
        return [], []

    return snippets, urls

@cached(ttl=None, cache=Cache.MEMORY)
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

@cached(ttl=None, cache=Cache.MEMORY)
async def get_responses(queries):
    tasks = [get_response(query) for query in queries]
    return await asyncio.gather(*tasks)


def clean_text(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace('-', ' ').replace(' .', '.')
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    return text

def refine_output(data):
    with st.expander("Source Excerpts:"):
        for text, info in sorted(data, key=lambda x: x[1]['score'], reverse=True)[:3]:
            st.write(f"Score: {info['score']}\n")
            cleaned_text = clean_text(text)
            
            # if "Table" in cleaned_text:
            #     st.write("Extracted Table:")
            #     st.write(create_table_from_text(cleaned_text))  # Example of integrating table extraction
            # else:
            st.write("Text:\n", cleaned_text)
            st.write("\n")



def process_data(data):
    # Sort the data based on the score in descending order and select the top three
    top_three = sorted(data, key=lambda x: x[1]['score'], reverse=True)[:3]
    
    # Format each text entry
    for text, info in top_three:
        cleaned_text = clean_text(text)
        st.write(f"Score: {info['score']}\nText: {cleaned_text}\n")

def embedchain_bot(db_path, api_key):
    return App.from_config(
        config={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o",
                    "temperature": 0.5,
                    "max_tokens": 4000,
                    "top_p": 1,
                    "stream": True,
                    "api_key": api_key,
                },
            },
            "vectordb": {
                "provider": "chroma",
                "config": {"collection_name": "chat-pdf", "dir": db_path, "allow_reset": True},
            },
            "embedder": {"provider": "openai", "config": {"api_key": api_key}},
            "chunker": {"chunk_size": 2000, "chunk_overlap": 0, "length_function": "len"},
        }
    )


def get_db_path():
    # tmpdirname = tempfile.mkdtemp()
    tmpdirname = tempfile.mkdtemp(prefix= "pdf_")
    return tmpdirname


def get_ec_app(api_key):
    if "app" in st.session_state:
        print("Found app in session state")
        app = st.session_state.app
    else:
        print("Creating app")
        db_path = get_db_path()
        app = embedchain_bot(db_path, api_key)
        st.session_state.app = app
    return app

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

@st.cache_data
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
    app = App()
    if "snippets" not in st.session_state:
        st.session_state["snippets"] = []
    if "urls" not in st.session_state:
        st.session_state["urls"] = []
    if "expert1_response" not in st.session_state:
        st.session_state["expert1_response"] = ""
    if "expert2_response" not in st.session_state:
        st.session_state["expert2_response"] = ""
    if "expert3_response" not in st.session_state:  
        st.session_state["expert3_response"] = ""
    if "sources" not in st.session_state:
        st.session_state["sources"] = []
    if "rag_response" not in st.session_state:
        st.session_state["rag_response"] = ""
        
    
    if check_password():
    
        # Obtain the initial query from the user
        original_query = st.text_input('Original Query', 'How long should alendronate be held for tooth extractions?')
        find_experts_messages = [{'role': 'system', 'content': system_prompt_expert_questions}, 
                                {'role': 'user', 'content': original_query}]
        
        site_number = st.number_input("Number of web pages to retrieve:", min_value=1, max_value=15, value=6, step=1)
        restrict_domains = st.checkbox("Restrict search to reliable medical domains", value=False)
        medical_domains = """site:www.nih.gov OR site:www.ncbi.nlm.nih.gov/books OR site:www.cdc.gov OR site:www.who.int OR site:www.pubmed.gov OR site:www.cochranelibrary.com OR 
    site:www.uptodate.com OR site:www.medscape.com OR site:www.ama-assn.org OR site:www.nejm.org OR 
    site:www.bmj.com OR site:www.thelancet.com OR site:www.jamanetwork.com OR site:www.mayoclinic.org OR site:www.acpjournals.org OR 
    site:www.cell.com OR site:www.nature.com OR site:www.springer.com OR site:www.wiley.com OR site:www.ahrq.gov OR site:www.edu"""
        if st.button('Begin Research'):
            
            # all_site_text = []
 

            

            if restrict_domains:
                domains = medical_domains
            else:
                domains = ""
                

            st.session_state.snippets, st.session_state.urls = realtime_search(original_query, domains, site_number)
            for site in st.session_state.urls:
                try:
                    app.add(site, data_type='web_page')
                    # st.session_state.search_results += f"{site}\n"
                    
                except Exception as e:
                    # st.error(f"Error adding {site}: {e}, skipping that one!")
                    st.sidebar.error(f"This site, {site}, won't let us retrieve content. Skipping it.")

            # web_results = app.query(original_query)
            # st.write(web_results)
            
            msg_placeholder = st.empty()
            msg_placeholder.markdown("Thinking...")
            full_response = ""

            q = queue.Queue()

            def app_response(result):
                st.write("Starting app_response function")
                llm_config = app.llm.config.as_dict()
                llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
                config = BaseLlmConfig(**llm_config)
                st.write("Before querying the app")
                st.write(f"App config: {app.llm.config.as_dict()}")
                st.write(f"App config: {app.llm.config.as_dict()}")
                answer, citations = app.query(f"Using only context, generate the best possible answer: {original_query}", config=config, citations=True)
                st.write("After querying the app")
                st.write(f"Answer: {answer}")
                st.write(f"Citations: {citations}")
                result["answer"] = answer
                result["citations"] = citations
                

            results = {}
            st.write("Starting thread for app_response")
            thread = threading.Thread(target=app_response, args=(results,))
            st.write("Before starting thread")
            thread.start()
            st.write("After starting thread")

            st.write("Before generating answer chunks")
            st.write(f"Queue size: {q.qsize()}")
            st.write(f"Queue size: {q.qsize()}")
            st.write("Before generating answer chunks")
            for answer_chunk in generate(q):
                st.write("Inside generate loop")
                st.write(f"Generated chunk: {answer_chunk}")
                st.write(f"Queue size after chunk: {q.qsize()}")
                st.write(f"Queue size after chunk: {q.qsize()}")
                st.write(f"Generated chunk: {answer_chunk}")
                full_response += answer_chunk
                msg_placeholder.markdown(full_response)
                st.write(f"Full response so far: {full_response}")

            st.write("Joining thread")
            st.write(f"Queue size before join: {q.qsize()}")
            thread.join()
            st.write("Thread joined")
            st.write(f"Queue size after join: {q.qsize()}")
            answer, citations = results["answer"], results["citations"]
            if citations:
                full_response += "\n\n**Sources**:\n"
                sources = []
                for i, citation in enumerate(citations):
                    source = citation[1]["url"]
                    pattern = re.compile(r"([^/]+)\.[^\.]+\.pdf$")
                    match = pattern.search(source)
                    if match:
                        source = match.group(1) + ".pdf"
                    sources.append(source)
                sources = list(set(sources))
                for source in sources:
                    full_response += f"- {source}\n"

            
            
            
            # *************************************************
            
            
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

        with st.sidebar:
            with st.expander("View Search Result Snippets"):
                if st.session_state.snippets:
                    for snippet in st.session_state.snippets:
                        snippet = snippet.replace('<END OF SITE>', '')
                        st.markdown(snippet)
                else:
                    st.markdown("No search results found!")

if __name__ == '__main__':
    main()

from hello import hello
