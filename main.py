from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class RetrievalAgent:
    def __init__(self):
        texts = [
            "私の趣味は読書です。",
            "私の好きな食べ物はカレーです。",
            "私の嫌いな食べ物は饅頭です。",
        ]

        vectorstore = FAISS.from_texts(
            texts, embedding=OpenAIEmbeddings(model="text-embedding-ada-002")
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        retriever_tool = create_retriever_tool(
            retriever,
            "search_about_me",
            "Searches and returns information about me.",
        )
        tools = [retriever_tool]

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

        agent = create_openai_tools_agent(llm, tools, prompt)

        self._agent_chain = AgentExecutor(agent=agent, tools=tools, memory=memory)

    def run(self, user_message: str) -> str:
        result = self._agent_chain.invoke({"input": user_message})
        return result["output"]


def main():
    agent = RetrievalAgent()
    ai_message = agent.run("私の趣味は何ですか？")
    print(ai_message)


if __name__ == "__main__":
    main()
