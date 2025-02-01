# Build Chatbot AI with LangChain

## Requirements
- Python 3.10+, the Python version for this project is 3.10.4
- Docker Desktop, [download](https://www.docker.com/products/docker-desktop/)
- OPENAI API KEY ([register](https://platform.openai.com/api-keys)) or DEEPSEEK API KEY ([register] (https://api-docs.deepseek.com/))
- Poetry, [install](https://python-poetry.org/docs/)

## Installation Steps

### Step 1: Setup Environment
- This project use Pyenv and Poetry to manage Python version and dependencies.
- Clone this repo to your local device, and run `poetry install` to install dependencies.

### Step 2: Setup Database (we use Milvus in this project)
- Open Docker Desktop
- In Terminal/ Command Prompt, run `docker compose up --build` to run initialize Milvus database
- Optional: To view the data in Milvus, we can install [Attus](https://milvus.io/docs/v2.0.x/attu_install-docker.md). Run below command:
```
docker run -p 8000:3000 -e HOST_URL=http://{ your machine IP }:8000 -e MILVUS_URL={your machine IP}:19530 zilliz/attu:latest
```
Replace {your machine IP} with your machine IP address. To check IP address, run `ifconfig en0` in MacOS or `ipconfig` in Windows.

### Step 3: Setup DEEPSEEK API KEY
- Create `.env` file
- Access to this [link]((https://api-docs.deepseek.com/)) to get DEEPSEEK API KEY
- Add below information to `.env` file:
```
DEEPSEEK_API_KEY = "your-DEEPSEEK-API-KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
```

### Step 4: Run the project
Clone this repo, at the root folder, run:
```python
poetry run streamlit run app.py
```