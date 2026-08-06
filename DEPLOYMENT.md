# AI Data Agent - Deployment & Optimization Guide

## Quick Start

### 1. Setup

```bash
# Clone or extract project
cd ai_data_agent

# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

### 2. Run with Docker (Recommended)

```bash
./start.sh docker

# Or manually:
docker-compose up -d
```

### 3. Run Locally

```bash
./start.sh local

# Or manually:
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Backend
python -m uvicorn app.main:app --reload --port 8000

# Terminal 3: Start Frontend
streamlit run app/frontend.py --server.port 8501
```

### 4. Access

- Frontend UI: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Performance Optimization

### RAG Pipeline Tuning

#### For Maximum Quality
```bash
# .env settings
EMBEDDING_MODEL=BAAI/bge-m3
USE_RERANKING=true
RETRIEVAL_CANDIDATES=30
RETRIEVAL_TOP_K=7
RERANK_THRESHOLD=0.5
USE_MULTI_QUERY=true
NUM_QUERY_VARIANTS=3
```

**Trade-offs:**
- ✅ Best retrieval accuracy
- ✅ Better handling of complex queries
- ❌ Slower (2-3x)
- ❌ Higher memory usage

#### For Maximum Speed
```bash
# .env settings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
USE_RERANKING=false
RETRIEVAL_CANDIDATES=10
RETRIEVAL_TOP_K=3
USE_MULTI_QUERY=false
```

**Trade-offs:**
- ✅ Very fast responses
- ✅ Low memory usage
- ❌ May miss relevant documents
- ❌ Less precise for complex queries

#### Balanced (Recommended)
```bash
# .env settings (default)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
USE_RERANKING=true
RETRIEVAL_CANDIDATES=20
RETRIEVAL_TOP_K=5
RERANK_THRESHOLD=0.35
USE_MULTI_QUERY=false
```

### Chunking Strategy

#### For Long Documents (reports, books)
```bash
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
MIN_CHUNK_SIZE=200
```

#### For Short Documents (emails, articles)
```bash
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MIN_CHUNK_SIZE=50
```

#### For Technical/Structured Content
```bash
CHUNK_SIZE=800
CHUNK_OVERLAP=150
MIN_CHUNK_SIZE=100
```

### Hybrid Search Weights

#### For Keyword-Heavy Queries (exact terms, names, codes)
```bash
BM25_WEIGHT=0.7
SEMANTIC_WEIGHT=0.3
```

#### For Conceptual Queries (meaning, themes)
```bash
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
```

#### Balanced
```bash
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
```

## Advanced Configuration

### LLM Selection

#### Production
```bash
DEFAULT_MODEL=gpt-5.2             # Best quality
FALLBACK_MODEL=gemini:gemini-flash-latest   # Cross-provider backup
LLM_TEMPERATURE=0.1               # Consistent, factual
```

#### Cost Optimization
```bash
DEFAULT_MODEL=gpt-5-mini          # Cheaper
FALLBACK_MODEL=gemini:gemini-flash-latest   # Cheap backup
LLM_TEMPERATURE=0.1
```

#### Creative Analysis
```bash
DEFAULT_MODEL=gpt-5.2
LLM_TEMPERATURE=0.3               # More creative
```

### Session Management

#### Short Sessions (chat-like)
```bash
SESSION_TTL=1800          # 30 minutes
MAX_CHAT_HISTORY=5
```

#### Long Sessions (research)
```bash
SESSION_TTL=7200          # 2 hours
MAX_CHAT_HISTORY=15
```

### File Upload Limits

```bash
# For large datasets
MAX_FILE_SIZE_MB=100

# For constrained environments
MAX_FILE_SIZE_MB=20
```

## Monitoring & Debugging

### Check System Status

```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# View logs (Docker)
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Common Issues

#### Out of Memory
**Symptoms:** Process crashes, slow responses
**Solutions:**
1. Reduce `CHUNK_SIZE` to 500-800
2. Decrease `RETRIEVAL_CANDIDATES` to 10-15
3. Use smaller embedding model
4. Disable reranking for large documents

#### Slow Retrieval
**Symptoms:** Long wait times for responses
**Solutions:**
1. Set `USE_RERANKING=false`
2. Reduce `RETRIEVAL_CANDIDATES` to 10
3. Use a smaller embedding model such as `BAAI/bge-base-en-v1.5` (768 dim)
4. Disable multi-query

#### Poor Retrieval Quality
**Symptoms:** Irrelevant documents retrieved
**Solutions:**
1. Enable reranking: `USE_RERANKING=true`
2. Increase `RERANK_THRESHOLD` to 0.4-0.5
3. Use the multilingual default: `BAAI/bge-m3` (1024 dim)
4. Adjust hybrid search weights
5. Increase `RETRIEVAL_CANDIDATES` to 25-30

#### Redis Connection Issues
```bash
# Check Redis
redis-cli ping

# Restart Redis
redis-server --daemonize yes

# Or with Docker
docker restart ai_agent_redis
```

## Production Deployment

### Environment Variables (Must Set)

```bash
# Required
OPENAI_API_KEY=sk-...

# Recommended
LOG_LEVEL=WARNING          # Reduce noise
SESSION_TTL=7200          # 2 hours
MAX_FILE_SIZE_MB=50
```

### Security Considerations

1. **API Keys**: Never commit `.env` to version control
2. **Redis**: Use password authentication in production
3. **File Upload**: Implement virus scanning for uploads
4. **Rate Limiting**: Add rate limits to endpoints
5. **HTTPS**: Use reverse proxy (nginx) with SSL

### Scaling

#### Horizontal Scaling
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      replicas: 3
    environment:
      - REDIS_URL=redis://redis:6379/0
```

#### Load Balancing
```nginx
# nginx.conf
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

### Resource Requirements

#### Minimum
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB

#### Recommended
- CPU: 4 cores
- RAM: 8GB
- Disk: 50GB

#### High Performance
- CPU: 8+ cores
- RAM: 16GB+
- Disk: 100GB+ SSD

## Customization

### Adding New File Formats

Edit `app/services/ingest.py`:

```python
def _parse_custom_format(self, filename, content, doc_id):
    # Your parsing logic
    text = extract_text(content)
    dataframes = extract_tables(content)
    return text, dataframes

# Register parser
parsers = {
    'custom': self._parse_custom_format,
    # ... existing parsers
}
```

### Custom Tools

Edit `app/services/tools.py`:

```python
async def execute_tool(tool_name, arguments, dataframes):
    if tool_name == "my_custom_tool":
        return await _my_custom_tool(arguments, dataframes)
    # ... existing tools
```

### Custom Prompts

Edit `app/utils/prompts.py`:

```python
def get_system_prompt(context, dataframes_info):
    prompt = f"""
    [Your custom instructions]
    
    Context: {context}
    Tables: {dataframes_info}
    """
    return prompt
```

## Best Practices

### Document Preparation

1. **Clean PDFs**: Remove headers/footers that repeat
2. **Table Extraction**: Use well-formatted tables
3. **File Naming**: Use descriptive names
4. **Organization**: Group related files in KB folder

### Query Formulation

**Good Queries:**
- "Show me revenue trends by quarter"
- "What are the top 5 products by sales?"
- "Summarize the risks mentioned in section 3"

**Poor Queries:**
- "Tell me about it" (too vague)
- "Everything about sales" (too broad)
- Very long multi-part questions (split them)

### Data Analysis

1. **Start Simple**: Begin with basic stats before complex viz
2. **Validate**: Check data types match expected operations
3. **Iterate**: Refine queries based on initial results
4. **Combine**: Mix document Q&A with data analysis

## Troubleshooting Checklist

- [ ] Redis is running (`redis-cli ping`)
- [ ] `.env` file exists with valid API key
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Ports 8000 and 8501 are available
- [ ] Sufficient disk space for embeddings cache
- [ ] Files in KB folder have supported extensions
- [ ] Check logs for specific errors

## Performance Benchmarks

Typical performance (MacBook Pro M1, 16GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Upload PDF (10 pages) | 2-3s | Including chunking & embedding |
| Upload Excel (1000 rows) | 1-2s | Including indexing |
| Simple query | 0.5-1s | Without reranking |
| Complex query | 1-3s | With reranking |
| Chart generation | 0.5-1s | Matplotlib rendering |
| Statistics calculation | 0.2-0.5s | Pandas operations |

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Review configuration in `.env`
3. Test API directly: `http://localhost:8000/docs`
4. Verify Redis: `redis-cli ping`
5. Check disk space and memory

## Maintenance

### Regular Tasks

```bash
# Clear Redis cache
redis-cli FLUSHDB

# Clean Docker volumes
docker-compose down -v

# Update dependencies
pip install -U -r requirements.txt
```

### Backup

```bash
# Backup KB folder
tar -czf kb_backup.tar.gz kb/

# Export Redis data
redis-cli BGSAVE
```
