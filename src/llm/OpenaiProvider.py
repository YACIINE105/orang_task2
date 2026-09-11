from langchain_openai import ChatOpenAI



class OpenAIProvider:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="your-api-key-here",  # or set OPENAI_API_KEY env var
        temperature=0.7,
    )

    response = llm.invoke("Hello, how are you?")
    print(response.content)









