<div align="center">
    <img src=“data/assets/logos/streamlit-app-logo.jpeg" width="50%">
</div>

# Conversational therapy chatbot using multi-agent LLM.

`Note : This solution is only a prototype. The chatbot can make mistakes. Please verify and double-check responses.`

## Summary
This repo implements a multi-agent chatbot system for emotional support of national service members in Singapore. 
The system consists of 4 agents controlled by a supervisor agent : 
1. Supervisor agent : Decides which agent to switch to based on the ongoing conversation 
2. Empathetic agent : Provides emotional support without any advice, tries to engage with the user about their problem until the user asks for advice.
3. Counsellor agent : Provides simple advice based on knowledge articles (using RAG) while sounding empathetic 
4. General agent : Provides answer to NS related queries (using RAG)
5. Survey agent : Administers CBT-based surveys to extract info about user's current state.

Other features : 
1. Memory : For multi-turn conversations
2. Events : Saves important reminders and events based on user utterances like "Remember to..." or "Remind me to..."
3. State switching : Ability to remember previous states and predict the next state for each agent switching

- Not implemented using langchain : Since this is only a prototype, we decided to implement the agentic system from scratch (without libraries like langchain or llamaindex). 
- Please also note that system artefacts (like data, prompts, schemas) are all stored in flat files for sake of simiplicity.

## Demo

[demo](https://drive.google.com/file/d/10B_2L3lH4-jZUyBks0FBr7QIbQu5z4WB/view?usp=sharing)

## Setup Virtual Environments

#### 1. Backend

For macOS (Apple Chip):
```bash
$ mamba env create -f dependencies/generative-ai-buddy-rev-conda-metal.yaml
```

For Linux, WSL, or macOS (Intel Chip):
```bash
$ mamba env create -f dependencies/generative-ai-buddy-rev-conda.yaml
```

#### 2. Add Submodules openwebui and pipelines (First time only)

```bash
$ git submodule add https://github.com/AI-DA-STC/openwebui-pipelines.git pipelines
```

```bash
$ git submodule add https://github.com/AI-DA-STC/open-webui.git open_webui
```

#### 3. Frontend

a. Install dependencies for openwebui 

Change directory to open_webui
```bash
$ cd open_webui
```

Copying required .env file
```bash
$ cp -RPp .env.example .env
```

Building Frontend Using Node
```bash
$ npm install
$ npm run build
```

Create and activate a Conda environment (openwebui strictly requires Python 3.11 to run, hence create a separate environment)
```bash
$ conda create --name open-webui-env python=3.11
$ conda activate open-webui-env
```

Install dependencies
```bash
$ pip install -r requirements.txt
```

b. Install dependencies for pipelines

Change directory to pipelines
```bash
$ cd ..
$ cd pipelines
```

Activate conda environment `open-webui-env` if not already
```bash
$ conda activate open-webui-env
$ pip install -r requirements.txt
```
## Environment variables 

Make sure to set the following environment variables in your `.env.*` file:
```bash
SUPABASE_URL=<SUPABASE_URL>
SUPABASE_KEY=<SUPABASE_KEY>
OPENAI_API_KEY=<OPENAI_API_KEY>
LANGCHAIN_API_KEY=<LANGCHAIN_API_KEY>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=BuddyAID
```

Before each script, export relevant environment variables through like so. To export variables listed within a file named (for example) `.env.dev`:

```bash
$ export $(cat .env.dev | xargs)
```

## Launch openwebui

Run buddyAID API, openwebui API, pipelines API and serve Redis respectively
```bash
$ python scripts/run_API.py

$ cd open_webui 
$ bash backend/start.sh

$ cd pipelines
$ bash start.sh

$ redis-server
```

## Repository Structure and Tree

### `generative-ai-buddy-rev/`
- `conf/` — Configuration files
- `core/` — Core application logic
  - `__init__.py`
  - `event_manager.py` - Manage events recorded from user convo
  - `logic.py` — Main application logic
  - `response_llm.py` — LLM response handling
  - `state.py` — Application state management
  - `utils.py` - Utility functions and helper scripts
- `data/` — Data storage
  - `assets/` — Static assets (images, etc.)
  - `logs/` - Conversation, error and temp user chat history
  - `processed/` - Final processed data
  - `prompts/` - all prompts used for counsellor,empathetic,RAG_NS, survey and metadata generation
  - `raw/` — Raw user data files 
  - `schema/` - processed data schema
  - `scraped/` - Scraped data files
- `data_processing/` — Data processing logic
  - `etl/` — Extract, Transform, Load processes, metadata generation
  - `scraping/` — Web scraping scripts and data
- `docs/` — Documentation
- `llms/` — All LLM-related functionality
  - `llm_counselor/` — Counselor-specific LLM functionality
  - `llm_empathetic/` — Empathetic response LLM functionality
  - `llm_rag/` — Retrieval-Augmented Generation LLM functionality
  - `llm_survey/` — Survey-related LLM functionality
- `reports/` — Generated reports
- `scripts/` — Standalone scripts
  - `load_index.py` — loading scraped to DBs for RAG app
  - `run_buddy_aid.py` — launch buddy aid app on terminal
- `semantic_layer/` — Semantic layer implementation
- `tests/` — Test files
  - `hallucination_test.py` 
  - `rag_test.py`
  - `safety_test.py`
  - `security_test.py`
- `ui/` — User Interface related code
  - `admin_ui/` — Admin interface
  - `user_reports_ui/` — User reports interface
  - `user_ui/` — Main user interface
