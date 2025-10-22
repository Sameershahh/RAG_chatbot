#  RAG Chatbot using FAISS, Sentence Transformers, and GPT-2

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using:
- **FAISS** for semantic vector search  
- **Sentence Transformers** for text embeddings  
- **GPT-2** for local text generation  
- **Streamlit** for a simple, interactive web interface  

You can query your **custom documents** (e.g., `.txt` files), and the chatbot will retrieve relevant text chunks and generate a context-aware answer.

---

##  Project Structure
```bash
├── .devcontainer/
│   └── devcontainer.json
├── docs/
│   ├── climate_change.txt
│   ├── global_warming.txt
│   ├── sample.txt
├── app.py                 # Streamlit UI for chatting with your RAG bot
├── indexer.py             # Creates FAISS index and metadata from docs/
├── rag.py                 # Retrieval + answer generation logic (backend module)
├── faiss.index            # Saved FAISS vector index
├── metadata.json          # Metadata with text chunks and sources
├── requirements.txt       # Dependencies list
```

##  Live Demo

 **Try it now:** [Live App](https://sameershah-chatbot.streamlit.app/)  

 

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Indexing Documents

Before chatting, you need to index your documents stored in the `docs/` folder.

Add your `.txt` files to the `docs/` directory and run:

```bash
python indexer.py
```
## Running the Chatbot
```bash
streamlit run app.py
```
## Example Query
### User:
What are the main causes of global warming?
### Chatbot:
Global warming is primarily caused by the increase of greenhouse gases such as carbon dioxide, methane, and nitrous oxide, which trap heat in the Earth's atmosphere...

## Author
**Sameer Shah** — AI & Full-Stack Developer  
[Portfolio](https://sameershah-portfolio.vercel.app/) 
