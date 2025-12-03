# Requirements Advisor MCP Server

An MCP (Model Context Protocol) server providing expert guidance on requirements management best practices from "The Essential Guide to Requirements Management and Traceability" by Jama Softare. A future release will add best practices from the INCOSE Guide and EARS documentation.

## Features

- **FastMCP Server**: Remote MCP server with Streamable HTTP transport, compatible with any LLM
- **Vector Search**: Semantic search over requirements management guidance
- **Multi-Source**: Supports multiple authoritative sources (Jama Guide, INCOSE, EARS)
- **Voyage AI Embeddings**: High-quality embeddings optimized for technical content
- **Docker Ready**: Containerized for easy deployment
- **Abstraction Layers**: Swap embedding providers or vector stores without code changes

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Voyage AI API key ([get one here](https://www.voyageai.com/))
- Content files (JSONL format) in the `content/` directory

### 1. Clone and Configure

```bash
cd requirements-advisor

# Create environment file
cp .env.example .env

# Edit .env and add your Voyage API key
nano .env  # or use your preferred editor
```

### 2. Add Content

Place your scraped content in the `content/` directory:
- `requirements_management_guide.jsonl` - Jama Guide (from jama-guide-scraper)
- `incose_gwr.jsonl` - INCOSE Guide (future)
- `ears_notation.jsonl` - EARS documentation (future)

### 3. Build and Run

```bash
# Build the container
docker compose build

# Ingest content into vector store
docker compose run --rm ingestion

# Start the server
docker compose up -d

# Check logs
docker compose logs -f mcp-server
```

The MCP server is now running at `http://localhost:8000/mcp`

---

## Deployment Scenarios

### Scenario A: Demo for Jama Executives

**Goal**: Quick demonstration of the MCP server capabilities

#### Option 1: Local Demo (Recommended for Execs)

Run on your laptop during the meeting:

```bash
# Ensure content is ingested
docker compose run --rm ingestion

# Start server
docker compose up

# Server available at http://localhost:8000/mcp
```

Connect with Claude Desktop or any MCP-compatible client (see "Connecting Clients" below).

#### Option 2: Cloud Demo (Shareable Link)

Deploy to a cloud provider for a persistent demo URL:

**Railway (Simplest)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Set environment variables in Railway dashboard
# Get public URL from Railway
```

**Render**
```bash
# Create render.yaml
# Push to GitHub
# Connect repo in Render dashboard
# Set VOYAGE_API_KEY in environment
```

**AWS/GCP/Azure**
```bash
# Build and push to container registry
docker build -t requirements-advisor .
docker tag requirements-advisor:latest <your-registry>/requirements-advisor:latest
docker push <your-registry>/requirements-advisor:latest

# Deploy to ECS/Cloud Run/Container Apps
```

For any cloud deployment, you'll need to:
1. Persist the ChromaDB volume or migrate to a managed vector store
2. Set `VOYAGE_API_KEY` as an environment variable/secret
3. Configure appropriate networking/firewall rules

---

### Scenario B: Local Development on Laptop

**Goal**: Run everything locally for development and testing

#### Using Docker (Recommended)

```bash
# 1. Start everything
docker compose up

# 2. In another terminal, test the search
docker compose exec mcp-server python -m requirements_advisor.cli test-search "how to write requirements"

# 3. Make changes and rebuild
docker compose up --build
```

#### Using UV (Without Docker)

```bash
# 1. Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv sync

# 3. Set up environment
cp .env.example .env
# Edit .env with your VOYAGE_API_KEY

# 4. Ingest content
uv run requirements-advisor ingest

# 5. Start server
uv run requirements-advisor serve

# 6. Or run directly with Python
source .venv/bin/activate
python -m requirements_advisor.cli serve
```

#### Using pip (Traditional)

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install package
pip install -e .

# 3. Configure and run
cp .env.example .env
# Edit .env
requirements-advisor ingest
requirements-advisor serve
```

---

## Connecting MCP Clients

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "requirements-advisor": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Restart Claude Desktop to connect.

### Claude Code

```bash
# Add the MCP server
claude mcp add requirements-advisor --url http://localhost:8000/mcp
```

### Other MCP Clients

Any MCP-compatible client can connect via Streamable HTTP transport at:
```
http://localhost:8000/mcp
```

For remote deployments, replace `localhost:8000` with your server URL.

---

## MCP Server Specification

### Server Metadata

| Property | Value |
|----------|-------|
| **Name** | `requirements-advisor` |
| **Transport** | Streamable HTTP (`/mcp` endpoint) |
| **Description** | Expert guidance on requirements management best practices. Provides answers from authoritative sources including Jama Software's Essential Guide to Requirements Management, INCOSE guidelines, and EARS notation. |

### Tools

This server exposes 4 tools. No resources or prompts are provided.

#### `search_requirements_guidance`

Search requirements management best practices and guidance.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `string` | Yes | - | Natural language question about requirements management |
| `top_k` | `integer` | No | `5` | Number of results to return (1-10) |
| `source` | `string` | No | `null` | Filter by source: `"jama_guide"`, `"incose"`, or `"ears"` |
| `include_images` | `boolean` | No | `true` | Include related images in response |

**Returns:** List of relevant guidance excerpts with source citations and optional images.

**Use for:** Writing requirements, traceability, validation/verification, regulatory compliance, systems engineering, industry-specific practices (medical, automotive, aerospace).

---

#### `get_definition`

Get the definition of a requirements management term or acronym.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `term` | `string` | Yes | - | The term or acronym to define |

**Returns:** Definition with source attribution.

**Use for:** Terms like SRS, EARS, Traceability, V&V, RTM, and other requirements management terminology.

---

#### `list_available_topics`

List the topics and sources available in the knowledge base.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| *(none)* | - | - | - | - |

**Returns:** Summary of available topics, sources, and document count.

**Use for:** Understanding what guidance is available before searching.

---

#### `get_best_practices`

Get best practices for a specific requirements management topic.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic` | `string` | Yes | - | The topic to get best practices for |
| `include_images` | `boolean` | No | `true` | Include related images in response |

**Returns:** Best practices with explanations, source citations, and optional images.

**Use for:** Topics like writing requirements, traceability, validation, change management, regulatory compliance, agile requirements.

---

### Example Queries

```
"How do I write good functional requirements?"
"What is requirements traceability and why does it matter?"
"Best practices for medical device requirements"
"Define EARS notation"
"What are non-functional requirements?"
```

---

## CLI Commands

```bash
# Start the MCP server
requirements-advisor serve [--host 0.0.0.0] [--port 8000]

# Ingest content into vector store
requirements-advisor ingest [--content-dir ./content] [--clear]

# Show configuration and status
requirements-advisor info

# Test a search query
requirements-advisor test-search "your query here" [--top-k 5]
```

---

## Project Structure

```
requirements-advisor/
├── pyproject.toml              # Python package configuration
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Multi-container orchestration
├── .env.example                # Environment template
├── src/requirements_advisor/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI commands
│   ├── config.py               # Pydantic settings
│   ├── logging.py              # Loguru logging configuration
│   ├── server.py               # FastMCP server + tools
│   ├── embeddings/
│   │   ├── base.py             # Abstract interface
│   │   └── voyage.py           # Voyage AI implementation
│   ├── vectorstore/
│   │   ├── base.py             # Abstract interface
│   │   └── chroma.py           # ChromaDB implementation
│   ├── images/
│   │   ├── base.py             # Image models (CachedImage, ImageIndex)
│   │   └── cache.py            # Image fetching and caching
│   └── ingestion/
│       └── pipeline.py         # Content ingestion
├── tests/                      # Test suite
│   ├── conftest.py             # Pytest fixtures
│   └── test_*.py               # Test modules
├── content/                    # JSONL content files
│   └── requirements_management_guide.jsonl
└── data/                       # Persistent data (gitignored)
    ├── chroma/                 # Vector store
    └── images/                 # Cached images
```

---

## Configuration

All configuration via environment variables (or `.env` file):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOYAGE_API_KEY` | ✅ | - | Voyage AI API key |
| `VOYAGE_MODEL` | | `voyage-context-3` | Embedding model (contextualized) |
| `VOYAGE_BATCH_SIZE` | | `20` | Texts per embedding API call |
| `VECTOR_STORE_TYPE` | | `chroma` | Vector store backend |
| `VECTOR_STORE_PATH` | | `./data/chroma` | Local storage path |
| `COLLECTION_NAME` | | `requirements_guidance` | Collection name |
| `CONTENT_DIR` | | `./content` | Content files location |
| `IMAGE_CACHE_PATH` | | `./data/images` | Image cache directory |
| `IMAGE_MAX_DIMENSION` | | `1024` | Max image dimension (pixels) |
| `IMAGE_QUALITY` | | `85` | JPEG compression quality |
| `IMAGE_FETCH_TIMEOUT` | | `30` | Image fetch timeout (seconds) |
| `HOST` | | `0.0.0.0` | Server bind host |
| `PORT` | | `8000` | Server bind port |
| `LOG_LEVEL` | | `INFO` | Logging level |
| `LOG_JSON` | | `false` | JSON log output format |

---

## Future Enhancements

- [ ] Qdrant vector store support for remote/managed deployment
- [ ] Additional embedding providers (OpenAI, Cohere)
- [ ] INCOSE Guide for Writing Requirements content
- [ ] EARS notation documentation
- [ ] Helm chart for Kubernetes deployment
- [ ] Authentication/API key support
- [ ] Usage analytics and feedback

---

## Troubleshooting

**"VOYAGE_API_KEY not set"**
- Ensure `.env` file exists with valid API key
- Check key is exported: `echo $VOYAGE_API_KEY`

**"No documents in vector store"**
- Run ingestion: `docker compose run --rm ingestion`
- Check content directory has JSONL files

**"Connection refused" from client**
- Ensure server is running: `docker compose ps`
- Check port 8000 is not blocked
- Verify URL in client config matches server

**Docker build fails**
- Ensure Docker is running
- Try: `docker compose build --no-cache`

---

## License

MIT License - See LICENSE file for details.
