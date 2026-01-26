from dotenv import load_dotenv
import operator
import datetime
from typing import Annotated, List, Literal, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from data import vector_store
load_dotenv()
#cai dat llm 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#khoi tao doi tuong memory
memory = MemorySaver()          
#class hoi thoai message
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

#Ham lay context moi nhat tu chatbot
def get_latest_tool_context(state: MessagesState):
    for msg in reversed(state["messages"]):
        if (hasattr(msg, 'type') and msg.type == "tool"):
            return msg.content, msg.name 
    return None, None
#Nut agent node chua 2 tool tra loi cau hoi
def agent_node(state: MessagesState):
    llm_with_tools = llm.bind_tools([retriever_tool, search_web])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

#Nut tra loi cau hoi
def generate_answer(state: MessagesState):
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get('role') == 'user')]
    latest_question = user_messages[-1].content if hasattr(user_messages[-1], 'content') else user_messages[-1].get('content', '')

    context, tool_name = get_latest_tool_context(state)
    
    if tool_name == "search_web":
        instruction = "Bạn đang sử dụng thông tin từ INTERNET. Hãy trả lời câu hỏi mới nhất ngắn gọn."
    elif tool_name == "retriever_tool":
        instruction = "Bạn đang sử dụng kiến thức Y TẾ NỘI BỘ trong vectore_store. Hãy trả lời câu hỏi mới nhất trong hợp lí, ngắn gọn."
    else:
        instruction = "Hãy trả lời câu hỏi một cách ngắn gọn."
    gen_prompt = f"""
    NHIỆM VỤ: {instruction}
    NGỮ CẢNH HIỆN TẠI: {context}
    CÂU HỎI HIỆN TẠI: {latest_question}
    
    LƯU Ý: Tuyệt đối không sử dụng thông tin từ các câu hỏi cũ trong lịch sử nếu không liên quan.
    """
    
    response = llm.invoke([HumanMessage(content=gen_prompt)])
    return {"messages": [response]}

#tool retrieve tu kho data
@tool()
def retriever_tool(query: str):
    """Truy xuất thông tin từ kho kiến thức Vinmec để trả lời câu hỏi."""

    retrieved_docs = vector_store.similarity_search(query, k=4)

    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata}\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tavily_tool = TavilySearch(
    max_results=1,
)
#tool search_web
@tool
def search_web(query: str) -> str:
    """
    Tìm kiếm thông tin trên mạng. Tool này có khả năng cập nhật tin nhắn 
    và thời gian thực (Real-time).
    """
    now = datetime.datetime.now()
    current_time_info = now.strftime("%A, ngày %d/%m/%Y (Giờ hệ thống: %H:%M)")
    
    enhanced_query = f"{query} (Ngày hiện tại: {current_time_info})"
    
    rall = tavily_tool.invoke(enhanced_query)
    results = rall.get("results")
    
    formatted_results = []
    for res in results:
        url = res.get('url')
        content = res.get('content')
        formatted_results.append(f"Source: {url}\nContent: {content}")

    return "\n\n".join(formatted_results)

#khoi tao ngay hom nay va system message
today = datetime.datetime.now().strftime("%A, %d/%m/%Y")
system_message = SystemMessage(content=f"""
Bạn là trợ lý y tế thông minh. Hôm nay là {today}.
Cách chọn công cụ:
1. Sử dụng 'retriever_tool' cho các câu hỏi về bệnh lý, triệu chứng, hoặc kiến thức y khoa nội bộ.
2. Sử dụng 'search_web' cho các tin tức mới, thời gian thực, hoặc khi dữ liệu nội bộ không có.
3. Nếu tự tin trả lời được ngay (như chào hỏi), không dùng tool.
4. Chỉ được sử dụng duy nhất 1 tool
""")

#tao graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("search_web", ToolNode([search_web]))
workflow.add_node("generate_answer", generate_answer)
#tao duong di nut dau tien
workflow.add_edge(START, "agent")
#ham quyet dinh se chon node nao
def route_decision(state: MessagesState) -> Literal["retrieve", "search_web", "end"]:
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return "end"
    
    tool_name = last_message.tool_calls[0]["name"]
    if tool_name == "retriever_tool":
        return "retrieve"
    else:
        return "search_web"
#tu nut agent se chon duong di qua node nao ke tiep
workflow.add_conditional_edges(
    "agent",
    route_decision,
    {
        "retrieve": "retrieve",
        "search_web": "search_web",
        "end": END
    }
)
#add edge
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("search_web", "generate_answer")
workflow.add_edge("generate_answer", END)
#compile
graph = workflow.compile(checkpointer=memory)
#config user voi thread
config = {"configurable":{"thread_id":"user_1"}}


