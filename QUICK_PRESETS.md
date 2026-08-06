# Quick Configuration Presets

## 🌍 Current Default: MULTILINGUAL (Maximum Accuracy)

The system is pre-configured for multilingual, maximum accuracy with these settings:

```bash
EMBEDDING_MODEL=BAAI/bge-m3              # 100+ languages
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-v2-m3   # Multilingual reranker
BM25_WEIGHT=0.3                          # Semantic-heavy
SEMANTIC_WEIGHT=0.7
```

---

## 📑 Content-Type Presets

### Research Papers & Books
```bash
# Preserve context, comprehensive retrieval
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
RETRIEVAL_TOP_K=7
RETRIEVAL_CANDIDATES=30
RERANK_THRESHOLD=0.35
```

### Articles & Reports (Default)
```bash
# Balanced approach
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
RETRIEVAL_CANDIDATES=25
RERANK_THRESHOLD=0.35
```

### Emails & Short Documents
```bash
# Quick retrieval, smaller chunks
CHUNK_SIZE=500
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=3
RETRIEVAL_CANDIDATES=15
RERANK_THRESHOLD=0.30
```

### Technical Documentation
```bash
# Preserve code blocks, structured content
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=5
RETRIEVAL_CANDIDATES=20
BM25_WEIGHT=0.5  # Balance keywords and semantics
SEMANTIC_WEIGHT=0.5
```

### Legal & Contracts
```bash
# Precise boundaries, high precision
CHUNK_SIZE=600
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=5
RETRIEVAL_CANDIDATES=20
RERANK_THRESHOLD=0.45
BM25_WEIGHT=0.5
SEMANTIC_WEIGHT=0.5
```

---

## 🎯 Performance Presets

### Maximum Quality (Current Default)
```bash
# Best accuracy, multilingual
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
USE_RERANKING=true
RETRIEVAL_CANDIDATES=25
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
DEFAULT_MODEL=gemini:gemini-2.0-flash-001
```
**Use when**: Quality is priority
**Speed**: Medium (~1.5s per query)

### Balanced
```bash
# Good quality, faster
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # English only
EMBEDDING_DIM=768
RERANKER_MODEL=BAAI/bge-reranker-large
USE_RERANKING=true
RETRIEVAL_CANDIDATES=20
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
DEFAULT_MODEL=gemini:gemini-2.0-flash-001
```
**Use when**: English-only documents, need speed
**Speed**: Fast (~0.8s per query)

### Maximum Speed
```bash
# Fastest, lower accuracy
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=384
USE_RERANKING=false
RETRIEVAL_CANDIDATES=10
RETRIEVAL_TOP_K=3
DEFAULT_MODEL=gemini:gemini-2.0-flash-lite-001
```
**Use when**: High-volume, latency-critical
**Speed**: Very fast (~0.3s per query)

---

## 🌐 Language-Specific Presets

### Multilingual (100+ Languages) - **CURRENT DEFAULT**
```bash
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
BM25_WEIGHT=0.3  # Favor semantics for multilingual
SEMANTIC_WEIGHT=0.7
```

### English Only (Best Performance)
```bash
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-large
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
```

### Chinese + English
```bash
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
BM25_WEIGHT=0.2  # Heavy semantic (handles both)
SEMANTIC_WEIGHT=0.8
```

---

## 🔧 Quick Switch Commands

### Switch to English-only (faster)
```bash
sed -i 's/EMBEDDING_MODEL=.*/EMBEDDING_MODEL=BAAI\/bge-base-en-v1.5/' .env
sed -i 's/EMBEDDING_DIM=.*/EMBEDDING_DIM=768/' .env
sed -i 's/BM25_WEIGHT=.*/BM25_WEIGHT=0.3/' .env
sed -i 's/SEMANTIC_WEIGHT=.*/SEMANTIC_WEIGHT=0.7/' .env
```

### Switch to research papers mode
```bash
sed -i 's/CHUNK_SIZE=.*/CHUNK_SIZE=1500/' .env
sed -i 's/CHUNK_OVERLAP=.*/CHUNK_OVERLAP=300/' .env
sed -i 's/RETRIEVAL_TOP_K=.*/RETRIEVAL_TOP_K=7/' .env
sed -i 's/RETRIEVAL_CANDIDATES=.*/RETRIEVAL_CANDIDATES=30/' .env
```

### Switch to maximum speed
```bash
sed -i 's/USE_RERANKING=.*/USE_RERANKING=false/' .env
sed -i 's/RETRIEVAL_CANDIDATES=.*/RETRIEVAL_CANDIDATES=10/' .env
sed -i 's/RETRIEVAL_TOP_K=.*/RETRIEVAL_TOP_K=3/' .env
```

### Back to multilingual defaults
```bash
sed -i 's/EMBEDDING_MODEL=.*/EMBEDDING_MODEL=BAAI\/bge-m3/' .env
sed -i 's/EMBEDDING_DIM=.*/EMBEDDING_DIM=1024/' .env
sed -i 's/RERANKER_MODEL=.*/RERANKER_MODEL=BAAI\/bge-reranker-v2-m3/' .env
sed -i 's/USE_RERANKING=.*/USE_RERANKING=true/' .env
```

---

## 📊 Model Performance Comparison

| Model | Languages | MTEB Score | Speed | Size | Best For |
|-------|-----------|------------|-------|------|----------|
| **BAAI/bge-m3** (Default) | 100+ | ~63.0 | Medium | 2.3GB | Multilingual |
| BAAI/bge-large-en-v1.5 | English | ~64.2 | Slow | 1.3GB | Best quality (EN) |
| BAAI/bge-base-en-v1.5 | English | ~63.6 | Fast | 400MB | Balanced (EN) |
| BAAI/bge-base-en-v1.5 | English | ~63.5 | Fast | 440MB | Speed priority |

| Reranker | Languages | Performance | Speed | Best For |
|----------|-----------|-------------|-------|----------|
| **BAAI/bge-reranker-v2-m3** (Default) | 100+ | Excellent | Medium | Multilingual |
| BAAI/bge-reranker-v2-gemma | Multi | Best | Slow | Max quality + GPU |
| BAAI/bge-reranker-large | EN/CN | Very Good | Fast | English-only |

---

## 💡 Recommendation by Use Case

| Use Case | Preset |
|----------|--------|
| **International documents** | Multilingual (current) |
| **Research/Academic** | Research Papers preset |
| **Business reports** | Articles & Reports (default) |
| **Customer emails** | Emails & Short Documents |
| **Code documentation** | Technical Documentation |
| **Legal contracts** | Legal & Contracts |
| **High-volume API** | Maximum Speed |
| **English-only production** | English Only preset |

---

## ⚠️ Important Notes

1. **After changing models**, first query will be slower (model download)
2. **GPU auto-detected** - no config needed
3. **Multilingual models** work great for English too
4. **Higher EMBEDDING_DIM** = more memory but better quality
5. **Restart required** after changing .env

---

**Current Configuration**: Multilingual, Maximum Accuracy
**Ready for**: 100+ languages, research papers, comprehensive analysis