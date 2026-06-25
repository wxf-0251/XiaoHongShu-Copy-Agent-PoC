# Enterprise-Marketing-Agent

## Quick Start

1. Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
uvicorn api.server:app --reload
```

4. Start the web UI:

```bash
streamlit run web/ui.py
```

## Notes

- The first run will automatically build `chroma_db_ollama/` from `data/my_knowledge.txt` if it does not exist.
- `HF_ENDPOINT` is optional.
