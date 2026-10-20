<p align="center">
  <a href="https://mnma.ai/" target="blank"><img src="assets/logo-full.svg" width="300" alt="MNMA Logo" /></a>
</p>

**Minima** is an open source RAG on-premises containers, with ability to integrate with ChatGPT and MCP.
Minima can also be used as a fully local RAG or with your own deployed LLM.

Minima currently supports four modes:
1. **Isolated installation (Ollama)** – Operate fully on-premises with containers, free from external dependencies such as ChatGPT or Claude. All neural networks (LLM, reranker, embedding) run on your cloud or PC, ensuring your data remains secure.

2. **Custom LLM (OpenAI-compatible API)** – Use your own deployed LLM with OpenAI-compatible API (vLLM, Ollama server, TGI, etc.). The indexer runs locally while the LLM can be on your server, cloud, or local machine. No Ollama deployment needed, lighter resource usage, and full control over your LLM infrastructure.

3. **Custom GPT** – Query your local documents using ChatGPT app or web with custom GPTs. The indexer runs on your cloud or local PC, while the primary LLM remains ChatGPT.

4. **Anthropic Claude** – Use Anthropic Claude app to query your local documents. The indexer operates on your local PC, while Anthropic Claude serves as the primary LLM.

---

## Running as Containers

### Quick Start with run.sh

The easiest way to start Minima is using the `run.sh` script:

```bash
./run.sh
```

You'll see the following options:
```
Select an option:
1) Fully Local Setup (Ollama)
2) Custom LLM (OpenAI-compatible API)
3) ChatGPT Integration
4) MCP usage
5) Quit
```

### Manual Docker Compose Commands

1. Create a .env file in the project's root directory (where you'll find .env.sample). Place .env in the same folder and copy all environment variables from .env.sample to .env.

2. Ensure your .env file includes the following variables:
<ul>
   <li> LOCAL_FILES_PATH </li>
   <li> EMBEDDING_MODEL_ID </li>
   <li> EMBEDDING_SIZE </li>
   <li> OLLAMA_MODEL (only for Ollama mode) </li>
   <li> RERANKER_MODEL (only for Ollama mode) </li>
   <li> LLM_BASE_URL (only for Custom LLM mode) </li>
   <li> LLM_MODEL (only for Custom LLM mode) </li>
   <li> LLM_API_KEY (optional for Custom LLM mode) </li>
   <li> USER_ID </li> - required for ChatGPT integration, just use your email
   <li> PASSWORD </li> - required for ChatGPT integration, just use any password
</ul>

3. For fully local installation use: **docker compose -f docker-compose-ollama.yml --env-file .env up --build**.

4. For custom LLM deployment (OpenAI-compatible API) use: **docker compose -f docker-compose-custom-llm.yml --env-file .env up --build**.

5. For ChatGPT enabled installation use: **docker compose -f docker-compose-chatgpt.yml --env-file .env up --build**.

6. For MCP integration (Anthropic Desktop app usage): **docker compose -f docker-compose-mcp.yml --env-file .env up --build**.

6. In case of ChatGPT enabled installation copy OTP from terminal where you launched docker and use [Minima GPT](https://chatgpt.com/g/g-r1MNTSb0Q-minima-local-computer-search)  

7. If you use Anthropic Claude, just add folliwing to **/Library/Application\ Support/Claude/claude_desktop_config.json**

```
{
    "mcpServers": {
      "minima": {
        "command": "uv",
        "args": [
          "--directory",
          "/path_to_cloned_minima_project/mcp-server",
          "run",
          "minima"
        ]
      }
    }
  }
```
   
8. To use fully local installation go to `cd electron`, then run `npm install` and `npm start` which will launch Minima electron app.

9. Ask anything, and you'll get answers based on local files in {LOCAL_FILES_PATH} folder.
---

## Variables Explained

**LOCAL_FILES_PATH**: Specify the root folder for indexing (on your cloud or local pc). Indexing is a recursive process, meaning all documents within subfolders of this root folder will also be indexed. Supported file types: .pdf, .xls, .docx, .txt, .md, .csv.

**EMBEDDING_MODEL_ID**: Specify the embedding model to use. Currently, only Sentence Transformer models are supported. Testing has been done with sentence-transformers/all-mpnet-base-v2, but other Sentence Transformer models can be used.

**EMBEDDING_SIZE**: Define the embedding dimension provided by the model, which is needed to configure Qdrant vector storage. Ensure this value matches the actual embedding size of the specified EMBEDDING_MODEL_ID.

**OLLAMA_MODEL**: Set up the Ollama model, use an ID available on the Ollama [site](https://ollama.com/search). Please, use LLM model here, not an embedding. This is only required when using Ollama (not needed when using custom LLM).

**LLM_BASE_URL**: (Optional) Base URL for your custom OpenAI-compatible LLM API endpoint. When this is set, Ollama will not be used and you don't need to deploy it.

**LLM_MODEL**: (Optional) Model name for your custom LLM. Required when LLM_BASE_URL is set.

**LLM_API_KEY**: (Optional) API key for your custom LLM. If your LLM doesn't require authentication, you can omit this or set it to any value.

**RERANKER_MODEL**: Specify the reranker model for Ollama mode. Currently, we have tested with BAAI rerankers. You can explore all available rerankers using this [link](https://huggingface.co/collections/BAAI/). **Note:** This is NOT required for Custom LLM mode - the reranker model will not be downloaded if you're using LLM_BASE_URL.

**USER_ID**: Just use your email here, this is needed to authenticate custom GPT to search in your data.

**PASSWORD**: Put any password here, this is used to create a firebase account for the email specified above.

---

## Examples

**Example of .env file for on-premises/local usage with Ollama:**
```
LOCAL_FILES_PATH=/Users/davidmayboroda/Downloads/PDFs/
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
EMBEDDING_SIZE=768
OLLAMA_MODEL=qwen2:0.5b # must be LLM model id from Ollama models page
RERANKER_MODEL=BAAI/bge-reranker-base # please, choose any BAAI reranker model
```

**Example of .env file for custom LLM deployment (OpenAI-compatible API):**
```
LOCAL_FILES_PATH=/Users/davidmayboroda/Downloads/PDFs/
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
EMBEDDING_SIZE=768
LLM_BASE_URL=http://your-llm-address:port/v1 # Your custom LLM endpoint
LLM_MODEL=Qwen/Qwen-1.7B # Your model name
LLM_API_KEY=not-needed # Optional: API key if required

# NOTE: OLLAMA_MODEL and RERANKER_MODEL are NOT needed for custom LLM mode
# The Docker build will skip reranker download automatically
```

**Important:** When using custom LLM mode, you do NOT need to set `OLLAMA_MODEL` or `RERANKER_MODEL` variables. The custom LLM workflow uses direct retrieval without reranking for better performance. The Dockerfile will automatically skip downloading the reranker model during build.

To use a chat ui, please navigate to **http://localhost:3000**

The custom LLM mode uses a different workflow compared to Ollama:

**Ollama Workflow:**
1. User query → Query enhancement (LLM call)
2. Document retrieval with reranking (HuggingFace CrossEncoder)
3. Answer generation (LLM call)

**Custom LLM Workflow:**
1. User query → LLM decides if document search is needed (function calling)
2. If needed: Direct vector search (no reranking)
3. LLM generates answer with or without retrieved context

**Compatible LLM Servers:**
- **vLLM** - High-performance inference server (`http://your-server:8000/v1`)
- **Text Generation Inference (TGI)** - Hugging Face's inference server
- **Ollama Server** - Ollama running in API mode
- **LiteLLM** - Proxy for multiple LLM providers
- **LocalAI** - OpenAI-compatible local inference
- **OpenAI API** - Directly use OpenAI's API
- **Any OpenAI-compatible endpoint**

This will automatically use `docker-compose-custom-llm.yml` which deploys only the necessary services (no Ollama container).

**Example of .env file for Claude app:**
```
LOCAL_FILES_PATH=/Users/davidmayboroda/Downloads/PDFs/
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
EMBEDDING_SIZE=768
```
For the Claude app, please apply the changes to the claude_desktop_config.json file as outlined above.

**To use MCP with GitHub Copilot:**
1. Create a .env file in the project’s root directory (where you’ll find env.sample). Place .env in the same folder and copy all environment variables from env.sample to .env.

2. Ensure your .env file includes the following variables:
    - LOCAL_FILES_PATH
    - EMBEDDING_MODEL_ID
    - EMBEDDING_SIZE
      
3. Create or update the `.vscode/mcp.json` with the following configuration:

````json
{
  "servers": {
    "minima": {
      "type": "stdio",
      "command": "path_to_cloned_minima_project/run_in_copilot.sh",
      "args": [
        "path_to_cloned_minima_project"
      ]
    }
  }
}
````

**Example of .env file for ChatGPT custom GPT usage:**
```
LOCAL_FILES_PATH=/Users/davidmayboroda/Downloads/PDFs/
EMBEDDING_MODEL_ID=sentence-transformers/all-mpnet-base-v2
EMBEDDING_SIZE=768
USER_ID=user@gmail.com # your real email
PASSWORD=password # you can create here password that you want
```

Also, you can run minima using **run.sh**.

---

**For MCP usage, please be sure that your local machines python is >=3.10 and 'uv' installed.**

Minima (https://github.com/dmayboroda/minima) is licensed under the Mozilla Public License v2.0 (MPLv2).

### Web sources in Custom LLM mode

The custom-LLM agent can search the web and read public pages with
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
Parallel is the default web provider, with no account or API key required;
anonymous search uses Fast mode. Web access requires explicit permission.
Keep it disabled when working with sensitive data or requiring local-only retrieval.

To allow web tools, add this to `.env` and recreate the custom-LLM containers:

```dotenv
MINIMA_ALLOW_WEB=true
```

Use the existing `docker compose -f docker-compose-custom-llm.yml --env-file .env up --build`
command, then ask a public-information question in the chat UI. The LLM can call
`web_search` followed by `web_fetch`, use the returned excerpts in its answer, and
return source URLs alongside document links. The model must support function calling.
Each question is an independent research run, matching the existing per-question
graph; related tool calls within that run share one opaque session ID.

Granting web permission allows model-generated queries and URLs to leave your
installation for `https://search.parallel.ai/mcp`. The model sees document context
and is instructed to keep private content out of web calls, but this is not a data
loss prevention boundary. Only enable it for data you permit the model to send
externally. Ollama, the Minima MCP server, and ChatGPT integration are unaffected.

Set `MINIMA_ALLOW_WEB=false` to revoke permission, or `MINIMA_WEB_SEARCH=none` to
disable the provider, then recreate the containers. Neither setting makes web
requests. An optional `PARALLEL_API_KEY` authenticates requests for higher limits;
a configured key is never silently replaced with anonymous access after failure.
Web calls have a 30-second timeout. Service errors, including free-tier limits,
are returned to the model as tool failures so it can explain missing evidence.

The web integration tests use a scripted model, the compiled agent graph and the
real MCP HTTP client against a deterministic HTTP fixture. They do not need model
weights, a GPU, Qdrant, an indexer, or inference credentials:

```bash
python -m venv /tmp/minima-web-tests
/tmp/minima-web-tests/bin/pip install -r llm/requirements-web-tests.txt
/tmp/minima-web-tests/bin/python -m unittest discover -s llm/tests -v
```
