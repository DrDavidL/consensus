import asyncio
import json
import re
import tempfile
from datetime import datetime

import aiohttp
import requests
import streamlit as st
from openai import OpenAI
import markdown2


from embedchain import App
from embedchain.config import BaseLlmConfig


from prompts import (
    system_prompt_expert_questions,
    expert1_system_prompt,
    expert2_system_prompt,
    expert3_system_prompt,
    optimize_search_terms_system_prompt,
)

# Set your OpenAI API key
api_key = st.secrets["OPENAI_API_KEY"]

# Function to replace the first user message
def replace_first_user_message(messages, new_message):
    for i, message in enumerate(messages):
        if message["role"] == "user":
            messages[i] = new_message
            break
        

def realtime_search(query, domains, max, start_year=2020):
    url = "https://real-time-web-search.p.rapidapi.com/search"
    full_query = f"{query} AND ({domains})"
    st.write(f'Full Query: {full_query}')
    
    # Define the start date and the current date
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Include the date range in the query string
    querystring = {
        "q": full_query,
        "limit": max,
        "from": start_date,
        "to": end_date
    }
    
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


# @cached(ttl=None, cache=Cache.MEMORY)
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

# @cached(ttl=None, cache=Cache.MEMORY)
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
        all_sources = ""
        for text, info in sorted(data, key=lambda x: x[1]['score'], reverse=True)[:3]:
            st.write(f"Score: {info['score']}\n")
            cleaned_text = clean_text(text)
            all_sources += cleaned_text
            # if "Table" in cleaned_text:
            #     st.write("Extracted Table:")
            #     st.write(create_table_from_text(cleaned_text))  # Example of integrating table extraction
            # else:
            st.write("Text:\n", cleaned_text)
            st.write("\n")
    return all_sources



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
    params = {k: v for k, v in params.items() if v != None}
    
    completion = client.chat.completions.create(**params)
    
    return completion

def check_password() -> bool:
    if "password" not in st.session_state:
        st.session_state.password = ""
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
    st.title('Helpful Answers with AI!')
    with st.expander("About this app"):
        st.info("""This app retrieves content from specific internet domains for an initial answer and asks AI personas their 
                opinions on the topic. Approaches shown to improve outputs like [chain of thought](https://arxiv.org/abs/2201.11903), 
                [expert rephrasing](https://arxiv.org/html/2311.04205v2), and [chain of verification](https://arxiv.org/abs/2309.11495)
                are applied to improve the quality of the responses and to reduce hallucination. Web sites are identified,processed and 
                content selectively retrieved for answers using [Real-Time Web Search](https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-web-search) 
                and the [EmbedChain](https://embedchain.ai/) library. The LLM model is [GPT-4o](https://openai.com/index/hello-gpt-4o/) from OpenAI.
                App author is David Liebovitz, MD
                """)
        site_number = st.number_input("Number of web pages to retrieve:", min_value=1, max_value=15, value=6, step=1)
        
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
    
    if "citation_data" not in st.session_state:
        st.session_state["citation_data"] = []
        
    if "source_chunks" not in st.session_state:
        st.session_state.source_chunks = ''
    
    if "experts" not in st.session_state:
        st.session_state.experts = []
        
    if "messages1" not in st.session_state:
        st.session_state.messages1 = []
        
    if "messages2" not in st.session_state:
        st.session_state.messages2 = []
    
    if "messages3" not in st.session_state:
        st.session_state.messages3 = []
        
    if "followup_messages" not in st.session_state:
        st.session_state.followup_messages = []
        
    if "expert_number" not in st.session_state:
        st.session_state.expert_number = 0
        
    if "expert_answers" not in st.session_state:
        st.session_state.expert_answers = []
        
    if "original_question" not in st.session_state:
        st.session_state.original_question = ""
        
    
    if check_password():
    
        # Obtain the initial query from the user
        original_query = st.text_input('Original Query', placeholder='Enter your question here...')
        st.session_state.original_question = original_query
        find_experts_messages = [{'role': 'system', 'content': system_prompt_expert_questions}, 
                                {'role': 'user', 'content': original_query}]
        

        # Define the domain strings
        medical_domains = """site:www.nih.gov OR site:www.ncbi.nlm.nih.gov/books OR site:www.cdc.gov OR site:www.who.int OR site:www.pubmed.gov OR site:www.cochranelibrary.com OR
        site:www.uptodate.com OR site:www.medscape.com OR site:www.ama-assn.org OR site:www.nejm.org OR
        site:www.bmj.com OR site:www.thelancet.com OR site:www.jamanetwork.com OR site:www.mayoclinic.org OR site:www.acpjournals.org OR
        site:www.cell.com OR site:www.nature.com OR site:www.springer.com OR site:www.wiley.com OR site:www.ahrq.gov OR site:www.nccn.org/guidelines/category_1 OR
        site:www.healthline.com OR site:www.medicalnewstoday.com OR site:www.webmd.com OR site:emedicine.medscape.com OR
        site:www.niddk.nih.gov OR site:kff.org OR site:academic.oup.com OR site:www.sciencedirect.com OR
        site:www.fda.gov OR site:www.ema.europa.eu OR site:clinicaltrials.gov OR site:drugs.com OR
        site:www.merckmanuals.com OR site:health.harvard.edu OR site:stanfordhealthcare.org OR site:clevelandclinic.org OR
        site:my.clevelandclinic.org"""

        reliable_domains = """site:www.cnn.com OR site:www.bbc.com OR site:www.npr.org OR site:www.reuters.com OR site:www.theguardian.com OR
        site:www.nytimes.com OR site:www.washingtonpost.com OR site:www.nbcnews.com OR site:www.cbsnews.com OR site:www.abcnews.go.com OR
        site:www.apnews.com OR site:www.bloomberg.com OR site:www.forbes.com OR site:www.nationalgeographic.com OR site:www.scientificamerican.com OR
        site:www.nature.com OR site:www.newscientist.com OR site:www.smithsonianmag.com OR site:www.wikipedia.org OR site:www.history.com OR
        site:www.britannica.com OR site:www.theatlantic.com OR site:www.vox.com OR site:www.propublica.org OR site:www.economist.com OR
        site:www.pbs.org OR site:www.nature.org OR site:www.academic.oup.com OR site:www.ted.com OR site:www.nasa.gov OR site:arxiv.org OR
        site:www.jstor.org OR site:scholar.google.com OR site:www.mit.edu OR site:www.stanford.edu OR site:www.harvard.edu OR
        site:www.yale.edu OR site:www.princeton.edu OR
        site:www.asahi.com OR site:www.ft.com OR site:www.wsj.com"""

        # Add radio buttons for domain selection
        restrict_domains = st.radio("Restrict Internet search domains to:", options=["Medical", "General Knowledge", "Full Internet", "No Internet"], horizontal=True)

        # Update the `domains` variable based on the selection
        if restrict_domains == "Medical":
            domains = medical_domains
        elif restrict_domains == "General Knowledge":
            domains = reliable_domains
        else:
            domains = ""  # Full Internet option doesn't restrict domains

        # Checkbox to reveal and edit domains
        if restrict_domains != "No Internet":
            edit_domains = st.checkbox("Reveal and Edit Selected Domains")

            # Display the selected domains in a text area if the checkbox is checked
            if edit_domains:
                domains = st.text_area("Edit domains (maintain format pattern):", domains, height=200)
        
        if st.button('Begin Research'):
            st.divider()
            app.reset()
            
            if restrict_domains != "No Internet":
            
                search_messages = [{'role': 'system', 'content': optimize_search_terms_system_prompt},
                                    {'role': 'user', 'content': original_query}]    
                response_google_search_terms = create_chat_completion(search_messages, temperature=0.3, )
                google_search_terms = response_google_search_terms.choices[0].message.content
                with st.spinner(f'Searching for "{google_search_terms}"...'):
                    st.session_state.snippets, st.session_state.urls = realtime_search(google_search_terms, domains, site_number)
                
                # Initialize a list to store blocked sites
                blocked_sites = []
                
                with st.spinner('Retrieving full content from web pages...'):
                    for site in st.session_state.urls:
                        try:
                            app.add(site, data_type='web_page')
                            
                        except Exception as e:
                            # Collect the blocked sites
                            blocked_sites.append(site)

                if blocked_sites:
                    with st.sidebar:
                        with st.expander("Sites Blocking Use"):
                            for site in blocked_sites:
                                st.error(f"This site, {site}, won't let us retrieve content. Skipping it.")


                llm_config = app.llm.config.as_dict()  
                config = BaseLlmConfig(**llm_config) 
                with st.spinner('Analyzing retrieved content...'):
                    try:                                                                                        
                        answer, citations = app.query(f"Using only context, provide the best possible answer to satisfy the user with the supportive evidence noted explicitly when possible: {original_query}", config=config, citations=True)                                               
                    except Exception as e:   
                        st.error(f"Error during app query: {e}")                                                                   
    
                full_response = ""
                if answer:                  
                    full_response = f"**Internet Based Response:** {answer} \n\n Search terms: {google_search_terms} \n\n"
                                    
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
                st.markdown(full_response)
                
                st.session_state.rag_response = full_response
                st.session_state.source_chunks = refine_output(citations)



            
            
            completion = create_chat_completion(messages=find_experts_messages, temperature=0.3, response_format="json_object")
            with st.sidebar:
                with st.expander("AI Personas Identified"):
                    # st.write(f"**Response:**")
                    json_output = completion.choices[0].message.content
                    # st.write(json_output)
                    experts, domains, expert_questions = extract_expert_info(json_output)
                    st.session_state.experts = experts
                    for expert in st.session_state.experts:
                        st.write(f"**{expert}**")
                    # st.write(f"**Experts:** {st.session_state.experts}")
                    # st.write(f"**Domains:** {domains}")
                    # st.write(f"**Expert Questions:** {expert_questions}")
            
            updated_expert1_system_prompt = expert1_system_prompt.format(expert=experts[0], domain=domains[0])
            updated_expert2_system_prompt = expert2_system_prompt.format(expert=experts[1], domain=domains[1])
            updated_expert3_system_prompt = expert3_system_prompt.format(expert=experts[2], domain=domains[2])
            updated_question1 = expert_questions[0]
            updated_question2 = expert_questions[1]
            updated_question3 = expert_questions[2]
            
            if restrict_domains != "No Internet":
                expert1_messages = [{'role': 'system', 'content': updated_expert1_system_prompt}, 
                                    {'role': 'user', 'content': updated_question1 + "Here's what I already found online: " + full_response}]
                st.session_state.messages1 = expert1_messages
                expert2_messages = [{'role': 'system', 'content': updated_expert2_system_prompt}, 
                                    {'role': 'user', 'content': updated_question2 + "Here's what I already found online: " + full_response}]
                st.session_state.messages2 = expert2_messages
                expert3_messages = [{'role': 'system', 'content': updated_expert3_system_prompt}, 
                                    {'role': 'user', 'content': updated_question3 + "Here's what I already found online: " + full_response}]
                st.session_state.messages3 = expert3_messages
                
            else:
                expert1_messages = [{'role': 'system', 'content': updated_expert1_system_prompt}, 
                                    {'role': 'user', 'content': updated_question1}]
                st.session_state.messages1 = expert1_messages
                expert2_messages = [{'role': 'system', 'content': updated_expert2_system_prompt}, 
                                    {'role': 'user', 'content': updated_question2}]
                st.session_state.messages2 = expert2_messages
                expert3_messages = [{'role': 'system', 'content': updated_expert3_system_prompt}, 
                                    {'role': 'user', 'content': updated_question3}]
                st.session_state.messages3 = expert3_messages
            
            with st.spinner('Waiting for experts to respond...'):
                st.session_state.expert_answers = asyncio.run(get_responses([expert1_messages, expert2_messages, expert3_messages]))

        if st.session_state.expert_answers:   
            with st.expander(f'AI {st.session_state.experts[0]} Perspective'):
                st.write(st.session_state.expert_answers[0]['choices'][0]['message']['content'])
                st.session_state.messages1.append({"role": "assistant", "content": st.session_state.expert_answers[0]['choices'][0]['message']['content']})
            with st.expander(f'AI {st.session_state.experts[1]} Perspective'):
                st.write(st.session_state.expert_answers[1]['choices'][0]['message']['content'])
                st.session_state.messages2.append({"role": "assistant", "content": st.session_state.expert_answers[1]['choices'][0]['message']['content']})
            with st.expander(f'AI {st.session_state.experts[2]} Perspective'):
                st.write(st.session_state.expert_answers[2]['choices'][0]['message']['content'])
                st.session_state.messages3.append({"role": "assistant", "content": st.session_state.expert_answers[2]['choices'][0]['message']['content']})
            


        
        if st.session_state.rag_response:            
            with st.sidebar:
                with st.expander("Web Response and Sources"):
                    st.write(st.session_state.rag_response)
                    st.write(st.session_state.source_chunks)
        
        if st.session_state.messages1:        
            if st.checkbox("Ask a Followup Question"):
                expert_chosen = st.selectbox("Choose an expert to ask a followup question:", st.session_state.experts)
                experts = st.session_state.experts
                if experts:
                    if expert_chosen == experts[0]:
                        st.session_state.followup_messages = st.session_state.messages1 
                        st.session_state.expert_number =1
                        
                    elif expert_chosen == experts[1]:
                        st.session_state.followup_messages = st.session_state.messages2 
                        st.session_state.expert_number =2
                        
                    elif expert_chosen == experts[2]:
                        st.session_state.followup_messages = st.session_state.messages3 
                        st.session_state.expert_number =3
                        
                
                if prompt := st.chat_input("Ask followup!"):
                    st.session_state.followup_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        client = OpenAI()
                        stream = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.followup_messages
                            ],
                            stream=True,
                        )
                        st.write(experts[st.session_state.expert_number-1] + ": ")
                        response = st.write_stream(stream)
                        st.session_state.followup_messages.append({"role": "assistant", "content": f"{experts[st.session_state.expert_number -1]}: {response}"})
                        full_conversation = ""
                        for message in st.session_state.followup_messages:
                            if message['role'] != 'system':
                                full_conversation += f"{message['role']}: {message['content']}\n\n"
                        
                        html = markdown2.markdown(full_conversation, extras=["tables"])
                        st.download_button('Download Followup Responses', html, f'followup_responses.html', 'text/html')
        
        if st.session_state.snippets:
            with st.sidebar:
                with st.expander("View Links from Internet Search"):
                    for snippet in st.session_state.snippets:
                        snippet = snippet.replace('<END OF SITE>', '')
                        st.markdown(snippet)
        
        
        if st.session_state.followup_messages:            
            with st.sidebar:
                with st.expander("Followup Conversation"):
                    full_conversation = ""
                    replace_first_user_message(st.session_state.followup_messages, {"role": "user", "content": st.session_state.original_question})
                    for message in st.session_state.followup_messages:
                        if message['role'] != 'system':
                            full_conversation += f"{message['role']}: {message['content']}\n\n"
                    full_conversation = full_conversation.replace("assistant:", "")
                    st.write(full_conversation)


if __name__ == '__main__':
    main()

from hello import hello
