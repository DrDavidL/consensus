## Clone, install requirements, set your own password, and obtain a Rapid Web Search API key and an OpenAI API key.

### Note main app is basic.py 

This app retrieves content from specific internet domains for an initial answer and asks AI personas their 
opinions on the topic. Approaches shown to improve outputs like [chain of thought](https://arxiv.org/abs/2201.11903), 
[expert rephrasing](https://arxiv.org/html/2311.04205v2), and [chain of verification](https://arxiv.org/abs/2309.11495)
are applied to improve the quality of the responses and to reduce hallucination. Web sites are identified,processed and 
content selectively retrieved for answers using [Real-Time Web Search](https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-web-search) 
and the [EmbedChain](https://embedchain.ai/) library. The LLM model is [GPT-4o](https://openai.com/index/hello-gpt-4o/) from OpenAI.
App author is David Liebovitz, MD

![alt text](<static/CleanShot 2024-06-09 at 22.10.39@2x.png>)