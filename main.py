from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def dummy():
    """Dummy tool that does nothing."""
    return "dummy"


tools = [dummy]


def create_agent():
    load_dotenv()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

    memory = ConversationBufferMemory(
        input_key="input", memory_key="chat_history", return_messages=True
    )

    agent = create_openai_functions_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, memory=memory)


def main():
    load_dotenv()

    agent_chain = create_agent()

    while True:
        user_message = input("You: ")
        result = agent_chain.invoke({"input": user_message})
        ai_message = result["output"]
        print(f"AI: {ai_message}")


if __name__ == "__main__":
    main()
