# Agentic RAG Medical Chatbot

An **Agentic RAG (Retrieval-Augmented Generation) chatbot** for medical information retrieval and conversational assistance. The system combines a **Vietnamese medical knowledge base from Vinmec** with **real-time web search** so that the chatbot can answer questions using internal medical documents when relevant and search the Internet when up-to-date information is required.

> **Disclaimer:** This project is intended for educational and information-retrieval purposes. It is not a substitute for diagnosis, treatment, or professional medical advice.

## Overview

The chatbot is designed to answer questions related to:

- Medical conditions and diseases
- Symptoms and common medical information
- Medical treatment and healthcare information available in the knowledge base
- Health-related questions that require current information from the Internet
- General conversational questions without unnecessary tool usage

The system does **not** simply send every question directly to an LLM. Instead, an agent determines which information source should be used:

1. **Internal medical knowledge base** → retrieve relevant Vinmec documents using vector similarity search.
2. **Internet search** → retrieve current information using Tavily when the question requires real-time or newer information, or when the internal knowledge base is insufficient.
3. **Direct LLM response** → used for simple questions such as greetings when external information is unnecessary.

Only **one retrieval/search tool is selected for each query**.

## Architecture

```text
User
  |
  v
Streamlit UI (gui.py)
  |
  v
LangGraph Agent
  |
  +-----------------------------+
  |                             |
  v                             v
Vinmec RAG                  Tavily Web Search
(retriever_tool)            (search_web)
  |                             |
  v                             v
Chroma Vector Store          Web Results
  |                             |
  +-------------+---------------+
                |
                v
        Answer Generation
        (Gemini 2.5 Flash)
                |
                v
          Final Response
```

## Main Components

### 1. LLM

The chatbot uses **Google Gemini 2.5 Flash** through LangChain:

```text
gemini-2.5-flash
```

The model is used for tool selection and final answer generation.

### 2. Agentic Workflow

The agent workflow is implemented with **LangGraph**.

The main workflow is:

```text
START
  -> agent
  -> retrieve OR search_web OR END
  -> generate_answer
  -> END
```

The agent decides whether to call `retriever_tool` or `search_web` based on the user's question.

### 3. Vinmec RAG Knowledge Base

The repository contains `Vinmec_output.json`, which stores structured medical articles and their text chunks.

The data pipeline in `data.py`:

1. Loads the JSON medical dataset.
2. Converts articles into LangChain `Document` objects.
3. Splits the documents into smaller chunks.
4. Generates embeddings using:

```text
models/text-embedding-004
```

5. Stores the embeddings in **ChromaDB**.
6. Uses similarity search to retrieve the top 4 relevant documents for a query.

The retriever is implemented as:

```python
vector_store.similarity_search(query, k=4)
```

Retrieved documents include metadata such as article title, category, URL, tags, and chunk information.

### 4. Web Search

The chatbot uses **Tavily Search** for Internet retrieval.

Web search is intended for:

- Current information
- Recent medical information
- Information that may not exist in the internal dataset
- Questions where real-time web sources are useful

The search result URL and content are passed to the answer-generation step.

### 5. Conversation Memory

The LangGraph workflow uses `MemorySaver` to maintain conversation state by thread.

The current configuration uses:

```python
thread_id = "user_1"
```

This allows the graph to maintain the conversation state across requests during execution.

## Project Structure

```text
Agentic-RAG-agent-chatbot/
│
├── agent.py              # LangGraph agent, tools and workflow
├── data.py               # Medical data loading, chunking and ChromaDB setup
├── gui.py                # Streamlit chatbot interface
├── Vinmec_output.json    # Vinmec medical knowledge base
├── main.ipynb            # Development / experimentation notebook
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Technologies

- **Python**
- **LangChain**
- **LangGraph**
- **Google Gemini 2.5 Flash**
- **Google Generative AI Embeddings**
- **ChromaDB**
- **Tavily Search**
- **Streamlit**
- **RAG (Retrieval-Augmented Generation)**
- **Agentic Workflow / Tool Calling**

## Requirements

Recommended environment:

- Python 3.10+
- Google Gemini API key
- Tavily API key
- Sufficient disk space for the local Chroma vector database

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nghductan/Agentic-RAG-agent-chatbot.git
cd Agentic-RAG-agent-chatbot
```

### 2. Create a virtual environment

Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure the environment contains the packages imported by `agent.py`, `data.py`, and `gui.py`, including LangChain, LangGraph, Chroma, Google Generative AI, Tavily, python-dotenv, and Streamlit.

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Do **not** commit the `.env` file or expose API keys publicly.

## Run the Chatbot

Start the Streamlit application:

```bash
streamlit run gui.py
```

After Streamlit starts, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How a Query Is Processed

Example user query:

```text
Trầm cảm có những triệu chứng phổ biến nào?
```

The processing flow is approximately:

```text
User Question
     |
     v
LangGraph Agent
     |
     v
Medical question?
     |
     v
retriever_tool
     |
     v
Chroma similarity search (top 4)
     |
     v
Relevant Vinmec documents
     |
     v
Gemini 2.5 Flash
     |
     v
Final answer
```

For a query requiring current information, for example:

```text
Có thông tin y tế mới nào liên quan đến ...?
```

the agent can select `search_web` instead:

```text
User Question
     |
     v
LangGraph Agent
     |
     v
search_web
     |
     v
Tavily Search
     |
     v
Web Results
     |
     v
Gemini 2.5 Flash
     |
     v
Final answer
```

## Key Features

- **Agentic tool selection** using LangGraph
- **RAG-based medical question answering** using Vinmec documents
- **Vector similarity search** with ChromaDB
- **Semantic embeddings** using Google's `text-embedding-004`
- **Real-time web search** through Tavily
- **LLM tool calling** for dynamic retrieval decisions
- **Conversation state management** with LangGraph `MemorySaver`
- **Streamlit chat interface** with processing status and conversation history
- **Source-aware retrieval**, preserving document metadata and web URLs

## Limitations

- The quality of answers depends on the quality and coverage of the Vinmec dataset and retrieved web results.
- Web search depends on Tavily availability and Internet access.
- The current ChromaDB persistence path is configured in `data.py` and may need to be changed for another machine/environment.
- The project is a prototype for learning and research and should not be used as a medical diagnostic system.

## Future Improvements

- Add multi-user session management instead of a fixed `thread_id`.
- Improve source citation and display in the Streamlit interface.
- Add a dedicated validation/safety agent for medical responses.
- Add conversation summarization for long conversations.
- Improve retrieval with hybrid search and reranking.
- Containerize the application with Docker.
- Add automated evaluation for RAG retrieval quality and answer quality.

## Author

**Duc Tan Nguyen**

GitHub: https://github.com/nghductan
