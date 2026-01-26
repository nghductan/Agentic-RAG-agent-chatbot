import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph, config 
from data import vector_store

# Cấu hình trang chuyên nghiệp
st.set_page_config(page_title="Chatbot", layout="wide")

# Sử dụng Custom CSS để làm UI đẹp hơn (Bo góc, đổ bóng)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 20px; border: 1px solid #ff4b4b; color: #ff4b4b; }
    .stStatus { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Điều khiển hệ thống")
    st.write("Trạng thái: **Sẵn sàng**")
    st.divider()
    
    if st.button("Xóa lịch sử hội thoại"):
        st.session_state.messages = []
        st.success("Đã làm mới cuộc hội thoại")
        st.rerun()
    
    st.info("Hệ thống Agentic RAG đang sử dụng dữ liệu nội bộ Vinmec và Search Internet.")

st.title("Chatbot tư vấn tâm lý")
st.caption("Ứng dụng trí tuệ nhân tạo trong hỗ trợ tra cứu bệnh lý và điều trị và có chức năng tìm kiếm thông tin trên web.")

if "messages" not in st.session_state:
    st.session_state.messages = []


chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)


user_input = st.chat_input("Nhập câu hỏi tại đây...")

if user_input:
    # 1. Hiển thị tin nhắn User
    st.session_state.messages.append(HumanMessage(content=user_input))
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Xử lý phản hồi từ Agent
        with st.chat_message("assistant"):
            final_answer = ""
            # Sử dụng trạng thái tiến trình (Status)
            with st.status("Đang phân tích và truy xuất dữ liệu...", expanded=True) as status:
                try:
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    for event in graph.stream(inputs, config=config, stream_mode="updates"):
                        for node_name, output in event.items():
                            status.write(f"Đã hoàn thành: {node_name}")
                            if "messages" in output:
                                # # Cập nhật kết quả cuối cùng từ Node generate_answer hoặc agent
                                # final_answer = output["messages"][-1].content
                                last_msg = output["messages"][-1]
                                if hasattr(last_msg, 'content'):
                                    final_answer = last_msg.content
                                elif isinstance(last_msg,dict):
                                    final_answer = last_msg.get('text',str(last_msg))
                                elif isinstance(last_msg,list):
                                    final_answer = last_msg[0].get('text')
                                else:
                                    final_answer = str(last_msg)

                    status.update(label="Xử lý hoàn tất", state="complete", expanded=False)
                
                except Exception as e:
                    status.update(label="Lỗi thực thi", state="error")
                    st.error(f"Chi tiết lỗi: {str(e)}")

            if final_answer:
                st.markdown(final_answer)
                st.session_state.messages.append(AIMessage(content=final_answer))
            else:
                st.warning("Hệ thống không tìm thấy câu trả lời phù hợp.")