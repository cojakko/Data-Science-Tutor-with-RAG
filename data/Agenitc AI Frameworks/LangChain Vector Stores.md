# LangChain overview

<Callout icon="bullhorn" color="#DFC5FE" iconType="regular">
  **LangChain v1.0 is now available!**

  For a complete list of changes and instructions on how to upgrade your code, see the [release notes](/oss/python/releases/langchain-v1) and [migration guide](/oss/python/migrate/langchain-v1).

  If you encounter any issues or have feedback, please [open an issue](https://github.com/langchain-ai/docs/issues/new?template=01-langchain.yml) so we can improve. To view v0.x documentation, [go to the archived content](https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs).
</Callout>

LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and [more](/oss/python/integrations/providers/overview). LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.

We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use [LangGraph](/oss/python/langgraph/overview), our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.

LangChain [agents](/oss/python/langchain/agents) are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

## <Icon icon="download" size={20} /> Install

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langchain
  ```

  ```bash uv theme={null}
  uv add langchain
  ```
</CodeGroup>

## <Icon icon="wand-magic-sparkles" /> Create an agent

```python  theme={null}
# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## <Icon icon="star" size={20} /> Core benefits

<Columns cols={2}>
  <Card title="Standard model interface" icon="arrows-rotate" href="/oss/python/langchain/models" arrow cta="Learn more">
    Different providers have unique APIs for interacting with models, including the format of responses. LangChain standardizes how you interact with models so that you can seamlessly swap providers and avoid lock-in.
  </Card>

  <Card title="Easy to use, highly flexible agent" icon="wand-magic-sparkles" href="/oss/python/langchain/agents" arrow cta="Learn more">
    LangChain's agent abstraction is designed to be easy to get started with, letting you build a simple agent in under 10 lines of code. But it also provides enough flexibility to allow you to do all the context engineering your heart desires.
  </Card>

  <Card title="Built on top of LangGraph" icon="circle-nodes" href="/oss/python/langgraph/overview" arrow cta="Learn more">
    LangChain's agents are built on top of LangGraph. This allows us to take advantage of LangGraph's durable execution, human-in-the-loop support, persistence, and more.
  </Card>

  <Card title="Debug with LangSmith" icon="eye" href="/langsmith/home" arrow cta="Learn more">
    Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
  </Card>
</Columns>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/overview.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>
Vector stores
A vector store stores embedded data and performs similarity search.

Pick your embedding model:
OpenAI
Azure
AWS
VertexAI
MistralAI
Cohere
Install dependencies
npm
yarn
pnpm
yarn add @langchain/openai

OPENAI_API_KEY=your-api-key

import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

Pick your vector store:
Memory
Chroma
FAISS
MongoDB
PGVector
Pinecone
Qdrant
Install dependencies
npm
yarn
pnpm
yarn add langchain

import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorStore = new MemoryVectorStore(embeddings);

LangChain.js integrates with a variety of vector stores. You can check out a full list below:

Name	Description
AnalyticDB	AnalyticDB for PostgreSQL is a massively parallel processing (MPP) da...
Astra DB	Only available on Node.js.
Azion EdgeSQL	The AzionVectorStore is used to manage and search through a collection
Azion EdgeSQL	The AzionVectorStore is used to manage and search through a collection
Azure AI Search	Azure AI Search (formerly known as Azure Search and Azure Cognitive S...
Azure Cosmos DB for MongoDB vCore	Azure Cosmos DB for MongoDB vCore makes it easy to create a database ...
Azure Cosmos DB for NoSQL	Azure Cosmos DB for NoSQL provides support for querying items with fl...
Cassandra	Only available on Node.js.
Chroma	Chroma is a AI-native
ClickHouse	Only available on Node.js.
CloseVector	available on both browser and Node.js
Cloudflare Vectorize	If you're deploying your project in a Cloudflare worker, you can use ...
Convex	LangChain.js supports Convex as a vector store, and supports the stan...
Couchbase	Couchbase is an award-winning distributed NoSQL cloud database that d...
Elasticsearch	Elasticsearch is a
Faiss	Faiss is a library for
Google Cloud SQL for PostgreSQL	Cloud SQL is a fully managed relational
Google Cloud SQL for PostgreSQL	Cloud SQL is a fully managed relational
Google Vertex AI Matching Engine	Only available on Node.js.
SAP HANA Cloud Vector Engine	SAP HANA Cloud Vector Engine is a vector store fully integrated into ...
HNSWLib	HNSWLib is an in-memory vector store that can be saved to a file. It
LanceDB	LanceDB is an embedded vector database for AI applications. It is ope...
libSQL	Turso is a SQLite-compatible database built on libSQL, the Open Contr...
MariaDB	This requires MariaDB 11.7 or later version
In-memory	LangChain offers is an in-memory, ephemeral vectorstore that stores
Milvus	Milvus is a vector database built for embeddings similarity search an...
Momento Vector Index (MVI)	MVI: the most productive, easiest to use, serverless vector index for...
MongoDB Atlas	This guide provides a quick overview for getting started with MongoDB
MyScale	Only available on Node.js.
Neo4j Vector Index	Neo4j is an open-source graph database with integrated support for ve...
Neon Postgres	Neon is a fully managed serverless PostgreSQL database. It separates ...
OpenSearch	Only available on Node.js.
PGVector	To enable vector search in generic PostgreSQL databases, LangChain.js
Pinecone	Pinecone is a vector database that helps
Prisma	For augmenting existing models in PostgreSQL database with vector sea...
Qdrant	Qdrant is a vector similarity search engine. It
Redis	Redis is a fast open source, in-memory data store.
Rockset	Rockset (acquired by OpenAI) is a real-time analyitics SQL database t...
SingleStore	SingleStoreDB is a robust, high-performance distributed SQL database ...
Supabase	Supabase is an open-source Firebase
Tigris	Tigris makes it easy to build AI applications with vector embeddings.
Turbopuffer	Setup
TypeORM	To enable vector search in a generic PostgreSQL database, LangChain.j...
Typesense	Vector store that utilizes the Typesense search engine.
Upstash Vector	Upstash Vector is a REST based serverless vector
USearch	Only available on Node.js.
Vectara	Vectara is a platform for building GenAI applications. It provides an...
Vercel Postgres	LangChain.js supports using the @vercel/postgres package to use gener...
Voy	Voy is a WASM vector similarity search engine written in Rust.
Weaviate	Weaviate is an open source vector database that
Xata	Xata is a serverless data platform, based on PostgreSQL. It provides ...
Zep Open Source	Zep is a long-term memory service for AI Assistant apps.
Zep Cloud	Zep is a long-term memory service for AI Assistant apps.
# LangChain overview

<Callout icon="bullhorn" color="#DFC5FE" iconType="regular">
  **LangChain v1.0 is now available!**

  For a complete list of changes and instructions on how to upgrade your code, see the [release notes](/oss/python/releases/langchain-v1) and [migration guide](/oss/python/migrate/langchain-v1).

  If you encounter any issues or have feedback, please [open an issue](https://github.com/langchain-ai/docs/issues/new?template=01-langchain.yml) so we can improve. To view v0.x documentation, [go to the archived content](https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs).
</Callout>

LangChain is the easiest way to start building agents and applications powered by LLMs. With under 10 lines of code, you can connect to OpenAI, Anthropic, Google, and [more](/oss/python/integrations/providers/overview). LangChain provides a pre-built agent architecture and model integrations to help you get started quickly and seamlessly incorporate LLMs into your agents and applications.

We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use [LangGraph](/oss/python/langgraph/overview), our low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.

LangChain [agents](/oss/python/langchain/agents) are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

## <Icon icon="download" size={20} /> Install

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langchain
  ```

  ```bash uv theme={null}
  uv add langchain
  ```
</CodeGroup>

## <Icon icon="wand-magic-sparkles" /> Create an agent

```python  theme={null}
# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

## <Icon icon="star" size={20} /> Core benefits

<Columns cols={2}>
  <Card title="Standard model interface" icon="arrows-rotate" href="/oss/python/langchain/models" arrow cta="Learn more">
    Different providers have unique APIs for interacting with models, including the format of responses. LangChain standardizes how you interact with models so that you can seamlessly swap providers and avoid lock-in.
  </Card>

  <Card title="Easy to use, highly flexible agent" icon="wand-magic-sparkles" href="/oss/python/langchain/agents" arrow cta="Learn more">
    LangChain's agent abstraction is designed to be easy to get started with, letting you build a simple agent in under 10 lines of code. But it also provides enough flexibility to allow you to do all the context engineering your heart desires.
  </Card>

  <Card title="Built on top of LangGraph" icon="circle-nodes" href="/oss/python/langgraph/overview" arrow cta="Learn more">
    LangChain's agents are built on top of LangGraph. This allows us to take advantage of LangGraph's durable execution, human-in-the-loop support, persistence, and more.
  </Card>

  <Card title="Debug with LangSmith" icon="eye" href="/langsmith/home" arrow cta="Learn more">
    Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
  </Card>
</Columns>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/overview.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>
# Vector stores

## Overview

A vector store stores [embedded](/oss/python/integrations/text_embedding) data and performs similarity search.

```mermaid  theme={null}
flowchart LR

    subgraph "📥 Indexing phase (store)"
        A[📄 Documents] --> B[🔢 Embedding model]
        B --> C[🔘 Embedding vectors]
        C --> D[(Vector store)]
    end

    subgraph "📤 Query phase (retrieval)"
        E[❓ Query text] --> F[🔢 Embedding model]
        F --> G[🔘 Query vector]
        G --> H[🔍 Similarity search]
        H --> D
        D --> I[📄 Top-k results]
    end
```

### Interface

LangChain provides a unified interface for vector stores, allowing you to:

* `add_documents` - Add documents to the store.
* `delete` - Remove stored documents by ID.
* `similarity_search` - Query for semantically similar documents.

This abstraction lets you switch between different implementations without altering your application logic.

### Initialization

To initialize a vector store, provide it with an embedding model:

```python  theme={null}
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embedding=SomeEmbeddingModel())
```

### Adding documents

Add [`Document`](https://reference.langchain.com/python/langchain_core/documents/#langchain_core.documents.base.Document) objects (holding `page_content` and optional metadata) like so:

```python  theme={null}
vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
```

### Deleting documents

Delete by specifying IDs:

```python  theme={null}
vector_store.delete(ids=["id1"])
```

### Similarity search

Issue a semantic query using `similarity_search`, which returns the closest embedded documents:

```python  theme={null}
similar_docs = vector_store.similarity_search("your query here")
```

Many vector stores support parameters like:

* `k` — number of results to return
* `filter` — conditional filtering based on metadata

### Similarity metrics & indexing

Embedding similarity may be computed using:

* **Cosine similarity**
* **Euclidean distance**
* **Dot product**

Efficient search often employs indexing methods such as HNSW (Hierarchical Navigable Small World), though specifics depend on the vector store.

### Metadata filtering

Filtering by metadata (e.g., source, date) can refine search results:

```python  theme={null}
vector_store.similarity_search(
  "query",
  k=3,
  filter={"source": "tweets"}
)
```

<important>
  Support for metadata-based filtering varies between implementations.
  Check the documentation of your chosen vector store for details.
</important>

## Top integrations

**Select embedding model:**

<AccordionGroup>
  <Accordion title="OpenAI">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-openai
      ```

      ```bash uv theme={null}
      uv add langchain-openai
      ```
    </CodeGroup>

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("OPENAI_API_KEY"):
      os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    ```
  </Accordion>

  <Accordion title="Azure">
    ```bash  theme={null}
    pip install -qU "langchain[azure]"
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
      os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

    from langchain_openai import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    ```
  </Accordion>

  <Accordion title="Google Gemini">
    ```bash  theme={null}
    pip install -qU langchain-google-genai
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
      os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    ```
  </Accordion>

  <Accordion title="Google Vertex">
    ```bash  theme={null}
    pip install -qU langchain-google-vertexai
    ```

    ```python  theme={null}
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model="text-embedding-005")
    ```
  </Accordion>

  <Accordion title="AWS">
    ```bash  theme={null}
    pip install -qU langchain-aws
    ```

    ```python  theme={null}
    from langchain_aws import BedrockEmbeddings

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    ```
  </Accordion>

  <Accordion title="HuggingFace">
    ```bash  theme={null}
    pip install -qU langchain-huggingface
    ```

    ```python  theme={null}
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    ```
  </Accordion>

  <Accordion title="Ollama">
    ```bash  theme={null}
    pip install -qU langchain-ollama
    ```

    ```python  theme={null}
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="llama3")
    ```
  </Accordion>

  <Accordion title="Cohere">
    ```bash  theme={null}
    pip install -qU langchain-cohere
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("COHERE_API_KEY"):
      os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

    from langchain_cohere import CohereEmbeddings

    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    ```
  </Accordion>

  <Accordion title="Mistral AI">
    ```bash  theme={null}
    pip install -qU langchain-mistralai
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("MISTRALAI_API_KEY"):
      os.environ["MISTRALAI_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")

    from langchain_mistralai import MistralAIEmbeddings

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    ```
  </Accordion>

  <Accordion title="Nomic">
    ```bash  theme={null}
    pip install -qU langchain-nomic
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("NOMIC_API_KEY"):
      os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter API key for Nomic: ")

    from langchain_nomic import NomicEmbeddings

    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    ```
  </Accordion>

  <Accordion title="NVIDIA">
    ```bash  theme={null}
    pip install -qU langchain-nvidia-ai-endpoints
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("NVIDIA_API_KEY"):
      os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
    ```
  </Accordion>

  <Accordion title="Voyage AI">
    ```bash  theme={null}
    pip install -qU langchain-voyageai
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("VOYAGE_API_KEY"):
      os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

    from langchain-voyageai import VoyageAIEmbeddings

    embeddings = VoyageAIEmbeddings(model="voyage-3")
    ```
  </Accordion>

  <Accordion title="IBM watsonx">
    ```bash  theme={null}
    pip install -qU langchain-ibm
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("WATSONX_APIKEY"):
      os.environ["WATSONX_APIKEY"] = getpass.getpass("Enter API key for IBM watsonx: ")

    from langchain_ibm import WatsonxEmbeddings

    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="<WATSONX PROJECT_ID>",
    )
    ```
  </Accordion>

  <Accordion title="Fake">
    ```bash  theme={null}
    pip install -qU langchain-core
    ```

    ```python  theme={null}
    from langchain_core.embeddings import DeterministicFakeEmbedding

    embeddings = DeterministicFakeEmbedding(size=4096)
    ```
  </Accordion>

  <Accordion title="xAI">
    ```bash  theme={null}
    pip install -qU "langchain[langchain-xai]"
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("XAI_API_KEY"):
      os.environ["XAI_API_KEY"] = getpass.getpass("Enter API key for xAI: ")

    from langchain.chat_models import init_chat_model

    model = init_chat_model("grok-2", model_provider="xai")
    ```
  </Accordion>

  <Accordion title="Perplexity">
    ```bash  theme={null}
    pip install -qU "langchain[langchain-perplexity]"
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("PPLX_API_KEY"):
      os.environ["PPLX_API_KEY"] = getpass.getpass("Enter API key for Perplexity: ")

    from langchain.chat_models import init_chat_model

    model = init_chat_model("llama-3.1-sonar-small-128k-online", model_provider="perplexity")
    ```
  </Accordion>

  <Accordion title="DeepSeek">
    ```bash  theme={null}
    pip install -qU "langchain[langchain-deepseek]"
    ```

    ```python  theme={null}
    import getpass
    import os

    if not os.environ.get("DEEPSEEK_API_KEY"):
      os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter API key for DeepSeek: ")

    from langchain.chat_models import init_chat_model

    model = init_chat_model("deepseek-chat", model_provider="deepseek")
    ```
  </Accordion>
</AccordionGroup>

**Select vector store:**

<AccordionGroup>
  <Accordion title="In-memory">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-core
      ```

      ```bash uv theme={null}
      uv add langchain-core
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)
    ```
  </Accordion>

  <Accordion title="Astra DB">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-astradb
      ```

      ```bash uv theme={null}
      uv add langchain-astradb
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_astradb import AstraDBVectorStore

    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="astra_vector_langchain",
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )
    ```
  </Accordion>

  <Accordion title="Azure Cosmos DB NoSQL">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-azure-ai azure-cosmos
      ```

      ```bash uv theme={null}
      uv add langchain-azure-ai
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
        AzureCosmosDBNoSqlVectorSearch,
    )
    vector_search = AzureCosmosDBNoSqlVectorSearch.from_documents(
        documents=docs,
        embedding=openai_embeddings,
        cosmos_client=cosmos_client,
        database_name=database_name,
        container_name=container_name,
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties={},
        full_text_search_enabled=True,
    )
    ```
  </Accordion>

  <Accordion title="Azure Cosmos DB Mongo vCore">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-azure-ai pymongo
      ```

      ```bash uv theme={null}
      uv add pymongo
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
        AzureCosmosDBMongoVCoreVectorSearch,
    )

    vectorstore = AzureCosmosDBMongoVCoreVectorSearch.from_documents(
        docs,
        openai_embeddings,
        collection=collection,
        index_name=INDEX_NAME,
    )
    ```
  </Accordion>

  <Accordion title="Chroma">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-chroma
      ```

      ```bash uv theme={null}
      uv add langchain-chroma
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    ```
  </Accordion>

  <Accordion title="FAISS">
    ```bash  theme={null}
    pip install -qU langchain-community
    ```

    ```python  theme={null}
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    ```
  </Accordion>

  <Accordion title="Milvus">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-milvus
      ```

      ```bash uv theme={null}
      uv add langchain-milvus
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_milvus import Milvus

    URI = "./milvus_example.db"

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    ```
  </Accordion>

  <Accordion title="MongoDB">
    ```bash  theme={null}
    pip install -qU langchain-mongodb
    ```

    ```python  theme={null}
    from langchain_mongodb import MongoDBAtlasVectorSearch

    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    ```
  </Accordion>

  <Accordion title="PGVector">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-postgres
      ```

      ```bash uv theme={null}
      uv add langchain-postgres
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_postgres import PGVector

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection="postgresql+psycopg://..."
    )
    ```
  </Accordion>

  <Accordion title="PGVectorStore">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-postgres
      ```

      ```bash uv theme={null}
      uv add langchain-postgres
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_postgres import PGEngine, PGVectorStore

    $engine = PGEngine.from_connection_string(
        url="postgresql+psycopg://..."
    )

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name='test_table',
        embedding_service=embedding
    )
    ```
  </Accordion>

  <Accordion title="Pinecone">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-pinecone
      ```

      ```bash uv theme={null}
      uv add langchain-pinecone
      ```
    </CodeGroup>

    ```python  theme={null}
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    pc = Pinecone(api_key=...)
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    ```
  </Accordion>

  <Accordion title="Qdrant">
    <CodeGroup>
      ```bash pip theme={null}
      pip install -qU langchain-qdrant
      ```

      ```bash uv theme={null}
      uv add langchain-qdrant
      ```
    </CodeGroup>

    ```python  theme={null}
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")

    vector_size = len(embeddings.embed_query("sample text"))

    if not client.collection_exists("test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=embeddings,
    )
    ```
  </Accordion>
</AccordionGroup>

| Vectorstore                                                                                                                                          | Delete by ID | Filtering | Search by Vector | Search with score | Async | Passes Standard Tests | Multi Tenancy | IDs in add Documents |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------- | ---------------- | ----------------- | ----- | --------------------- | ------------- | -------------------- |
| [`AstraDBVectorStore`](/oss/python/integrations/vectorstores/astradb)                                                                                | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ✅             | ✅                    |
| [`AzureCosmosDBNoSqlVectorStore`](/oss/python/integrations/vectorstores/azure_cosmos_db_no_sql)                                                      | ✅            | ✅         | ✅                | ✅                 | ❌     | ✅                     | ✅             | ✅                    |
| [`AzureCosmosDBMongoVCoreVectorStore`](/oss/python/integrations/vectorstores/azure_cosmos_db_mongo_vcore)                                            | ✅            | ✅         | ✅                | ✅                 | ❌     | ✅                     | ✅             | ✅                    |
| [`Chroma`](/oss/python/integrations/vectorstores/chroma)                                                                                             | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ✅             | ✅                    |
| [`Clickhouse`](/oss/python/integrations/vectorstores/clickhouse)                                                                                     | ✅            | ✅         | ❌                | ✅                 | ❌     | ❌                     | ❌             | ✅                    |
| [`CouchbaseSearchVectorStore`](/oss/python/integrations/vectorstores/couchbase)                                                                      | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ✅             | ✅                    |
| [`DatabricksVectorSearch`](/oss/python/integrations/vectorstores/databricks_vector_search)                                                           | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ❌             | ✅                    |
| [`ElasticsearchStore`](/oss/python/integrations/vectorstores/elasticsearch)                                                                          | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ❌             | ✅                    |
| [`FAISS`](/oss/python/integrations/vectorstores/faiss)                                                                                               | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ❌             | ✅                    |
| [`InMemoryVectorStore`](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html) | ✅            | ✅         | ❌                | ✅                 | ✅     | ❌                     | ❌             | ✅                    |
| [`Milvus`](/oss/python/integrations/vectorstores/milvus)                                                                                             | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ✅             | ✅                    |
| [`Moorcheh`](/oss/python/integrations/vectorstores/moorcheh)                                                                                         | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ✅             | ✅                    |
| [`MongoDBAtlasVectorSearch`](/oss/python/integrations/vectorstores/mongodb_atlas)                                                                    | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ✅             | ✅                    |
| [`openGauss`](/oss/python/integrations/vectorstores/opengauss)                                                                                       | ✅            | ✅         | ✅                | ✅                 | ❌     | ✅                     | ❌             | ✅                    |
| [`PGVector`](/oss/python/integrations/vectorstores/pgvector)                                                                                         | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ❌             | ✅                    |
| [`PGVectorStore`](/oss/python/integrations/vectorstores/pgvectorstore)                                                                               | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ❌             | ✅                    |
| [`PineconeVectorStore`](/oss/python/integrations/vectorstores/pinecone)                                                                              | ✅            | ✅         | ✅                | ❌                 | ✅     | ❌                     | ❌             | ✅                    |
| [`QdrantVectorStore`](/oss/python/integrations/vectorstores/qdrant)                                                                                  | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ✅             | ✅                    |
| [`Weaviate`](/oss/python/integrations/vectorstores/weaviate)                                                                                         | ✅            | ✅         | ✅                | ✅                 | ✅     | ❌                     | ✅             | ✅                    |
| [`SQLServer`](/oss/python/integrations/vectorstores/sqlserver)                                                                                       | ✅            | ✅         | ✅                | ✅                 | ❌     | ❌                     | ❌             | ✅                    |
| [`ZeusDB`](/oss/python/integrations/vectorstores/zeusdb)                                                                                             | ✅            | ✅         | ✅                | ✅                 | ✅     | ✅                     | ❌             | ✅                    |

## All vector stores

<Columns cols={3}>
  <Card title="Activeloop Deep Lake" icon="link" href="/oss/python/integrations/vectorstores/activeloop_deeplake" arrow="true" cta="View guide" />

  <Card title="Alibaba Cloud OpenSearch" icon="link" href="/oss/python/integrations/vectorstores/alibabacloud_opensearch" arrow="true" cta="View guide" />

  <Card title="AnalyticDB" icon="link" href="/oss/python/integrations/vectorstores/analyticdb" arrow="true" cta="View guide" />

  <Card title="Annoy" icon="link" href="/oss/python/integrations/vectorstores/annoy" arrow="true" cta="View guide" />

  <Card title="Apache Doris" icon="link" href="/oss/python/integrations/vectorstores/apache_doris" arrow="true" cta="View guide" />

  <Card title="ApertureDB" icon="link" href="/oss/python/integrations/vectorstores/aperturedb" arrow="true" cta="View guide" />

  <Card title="Astra DB Vector Store" icon="link" href="/oss/python/integrations/vectorstores/astradb" arrow="true" cta="View guide" />

  <Card title="Atlas" icon="link" href="/oss/python/integrations/vectorstores/atlas" arrow="true" cta="View guide" />

  <Card title="AwaDB" icon="link" href="/oss/python/integrations/vectorstores/awadb" arrow="true" cta="View guide" />

  <Card title="Azure Cosmos DB Mongo vCore" icon="link" href="/oss/python/integrations/vectorstores/azure_cosmos_db_mongo_vcore" arrow="true" cta="View guide" />

  <Card title="Azure Cosmos DB No SQL" icon="link" href="/oss/python/integrations/vectorstores/azure_cosmos_db_no_sql" arrow="true" cta="View guide" />

  <Card title="Azure AI Search" icon="link" href="/oss/python/integrations/vectorstores/azuresearch" arrow="true" cta="View guide" />

  <Card title="Bagel" icon="link" href="/oss/python/integrations/vectorstores/bagel" arrow="true" cta="View guide" />

  <Card title="BagelDB" icon="link" href="/oss/python/integrations/vectorstores/bageldb" arrow="true" cta="View guide" />

  <Card title="Baidu Cloud ElasticSearch VectorSearch" icon="link" href="/oss/python/integrations/vectorstores/baiducloud_vector_search" arrow="true" cta="View guide" />

  <Card title="Baidu VectorDB" icon="link" href="/oss/python/integrations/vectorstores/baiduvectordb" arrow="true" cta="View guide" />

  <Card title="Apache Cassandra" icon="link" href="/oss/python/integrations/vectorstores/cassandra" arrow="true" cta="View guide" />

  <Card title="Chroma" icon="link" href="/oss/python/integrations/vectorstores/chroma" arrow="true" cta="View guide" />

  <Card title="Clarifai" icon="link" href="/oss/python/integrations/vectorstores/clarifai" arrow="true" cta="View guide" />

  <Card title="ClickHouse" icon="link" href="/oss/python/integrations/vectorstores/clickhouse" arrow="true" cta="View guide" />

  <Card title="Couchbase" icon="link" href="/oss/python/integrations/vectorstores/couchbase" arrow="true" cta="View guide" />

  <Card title="DashVector" icon="link" href="/oss/python/integrations/vectorstores/dashvector" arrow="true" cta="View guide" />

  <Card title="Databricks" icon="link" href="/oss/python/integrations/vectorstores/databricks_vector_search" arrow="true" cta="View guide" />

  <Card title="IBM Db2" icon="link" href="/oss/python/integrations/vectorstores/db2" arrow="true" cta="View guide" />

  <Card title="DingoDB" icon="link" href="/oss/python/integrations/vectorstores/dingo" arrow="true" cta="View guide" />

  <Card title="DocArray HnswSearch" icon="link" href="/oss/python/integrations/vectorstores/docarray_hnsw" arrow="true" cta="View guide" />

  <Card title="DocArray InMemorySearch" icon="link" href="/oss/python/integrations/vectorstores/docarray_in_memory" arrow="true" cta="View guide" />

  <Card title="Amazon Document DB" icon="link" href="/oss/python/integrations/vectorstores/documentdb" arrow="true" cta="View guide" />

  <Card title="DuckDB" icon="link" href="/oss/python/integrations/vectorstores/duckdb" arrow="true" cta="View guide" />

  <Card title="China Mobile ECloud ElasticSearch" icon="link" href="/oss/python/integrations/vectorstores/ecloud_vector_search" arrow="true" cta="View guide" />

  <Card title="Elasticsearch" icon="link" href="/oss/python/integrations/vectorstores/elasticsearch" arrow="true" cta="View guide" />

  <Card title="Epsilla" icon="link" href="/oss/python/integrations/vectorstores/epsilla" arrow="true" cta="View guide" />

  <Card title="Faiss" icon="link" href="/oss/python/integrations/vectorstores/faiss" arrow="true" cta="View guide" />

  <Card title="Faiss (Async)" icon="link" href="/oss/python/integrations/vectorstores/faiss_async" arrow="true" cta="View guide" />

  <Card title="FalkorDB" icon="link" href="/oss/python/integrations/vectorstores/falkordbvector" arrow="true" cta="View guide" />

  <Card title="Gel" icon="link" href="/oss/python/integrations/vectorstores/gel" arrow="true" cta="View guide" />

  <Card title="Google AlloyDB" icon="link" href="/oss/python/integrations/vectorstores/google_alloydb" arrow="true" cta="View guide" />

  <Card title="Google BigQuery Vector Search" icon="link" href="/oss/python/integrations/vectorstores/google_bigquery_vector_search" arrow="true" cta="View guide" />

  <Card title="Google Cloud SQL for MySQL" icon="link" href="/oss/python/integrations/vectorstores/google_cloud_sql_mysql" arrow="true" cta="View guide" />

  <Card title="Google Cloud SQL for PostgreSQL" icon="link" href="/oss/python/integrations/vectorstores/google_cloud_sql_pg" arrow="true" cta="View guide" />

  <Card title="Firestore" icon="link" href="/oss/python/integrations/vectorstores/google_firestore" arrow="true" cta="View guide" />

  <Card title="Google Memorystore for Redis" icon="link" href="/oss/python/integrations/vectorstores/google_memorystore_redis" arrow="true" cta="View guide" />

  <Card title="Google Spanner" icon="link" href="/oss/python/integrations/vectorstores/google_spanner" arrow="true" cta="View guide" />

  <Card title="Google Vertex AI Feature Store" icon="link" href="/oss/python/integrations/vectorstores/google_vertex_ai_feature_store" arrow="true" cta="View guide" />

  <Card title="Google Vertex AI Vector Search" icon="link" href="/oss/python/integrations/vectorstores/google_vertex_ai_vector_search" arrow="true" cta="View guide" />

  <Card title="Hippo" icon="link" href="/oss/python/integrations/vectorstores/hippo" arrow="true" cta="View guide" />

  <Card title="Hologres" icon="link" href="/oss/python/integrations/vectorstores/hologres" arrow="true" cta="View guide" />

  <Card title="Jaguar Vector Database" icon="link" href="/oss/python/integrations/vectorstores/jaguar" arrow="true" cta="View guide" />

  <Card title="Kinetica" icon="link" href="/oss/python/integrations/vectorstores/kinetica" arrow="true" cta="View guide" />

  <Card title="LanceDB" icon="link" href="/oss/python/integrations/vectorstores/lancedb" arrow="true" cta="View guide" />

  <Card title="Lantern" icon="link" href="/oss/python/integrations/vectorstores/lantern" arrow="true" cta="View guide" />

  <Card title="Lindorm" icon="link" href="/oss/python/integrations/vectorstores/lindorm" arrow="true" cta="View guide" />

  <Card title="LLMRails" icon="link" href="/oss/python/integrations/vectorstores/llm_rails" arrow="true" cta="View guide" />

  <Card title="ManticoreSearch" icon="link" href="/oss/python/integrations/vectorstores/manticore_search" arrow="true" cta="View guide" />

  <Card title="MariaDB" icon="link" href="/oss/python/integrations/vectorstores/mariadb" arrow="true" cta="View guide" />

  <Card title="Marqo" icon="link" href="/oss/python/integrations/vectorstores/marqo" arrow="true" cta="View guide" />

  <Card title="Meilisearch" icon="link" href="/oss/python/integrations/vectorstores/meilisearch" arrow="true" cta="View guide" />

  <Card title="Amazon MemoryDB" icon="link" href="/oss/python/integrations/vectorstores/memorydb" arrow="true" cta="View guide" />

  <Card title="Milvus" icon="link" href="/oss/python/integrations/vectorstores/milvus" arrow="true" cta="View guide" />

  <Card title="Momento Vector Index" icon="link" href="/oss/python/integrations/vectorstores/momento_vector_index" arrow="true" cta="View guide" />

  <Card title="Moorcheh" icon="link" href="/oss/python/integrations/vectorstores/moorcheh" arrow="true" cta="View guide" />

  <Card title="MongoDB Atlas" icon="link" href="/oss/python/integrations/vectorstores/mongodb_atlas" arrow="true" cta="View guide" />

  <Card title="MyScale" icon="link" href="/oss/python/integrations/vectorstores/myscale" arrow="true" cta="View guide" />

  <Card title="Neo4j Vector Index" icon="link" href="/oss/python/integrations/vectorstores/neo4jvector" arrow="true" cta="View guide" />

  <Card title="NucliaDB" icon="link" href="/oss/python/integrations/vectorstores/nucliadb" arrow="true" cta="View guide" />

  <Card title="Oceanbase" icon="link" href="/oss/python/integrations/vectorstores/oceanbase" arrow="true" cta="View guide" />

  <Card title="openGauss" icon="link" href="/oss/python/integrations/vectorstores/opengauss" arrow="true" cta="View guide" />

  <Card title="OpenSearch" icon="link" href="/oss/python/integrations/vectorstores/opensearch" arrow="true" cta="View guide" />

  <Card title="Oracle AI Vector Search" icon="link" href="/oss/python/integrations/vectorstores/oracle" arrow="true" cta="View guide" />

  <Card title="Pathway" icon="link" href="/oss/python/integrations/vectorstores/pathway" arrow="true" cta="View guide" />

  <Card title="Postgres Embedding" icon="link" href="/oss/python/integrations/vectorstores/pgembedding" arrow="true" cta="View guide" />

  <Card title="PGVecto.rs" icon="link" href="/oss/python/integrations/vectorstores/pgvecto_rs" arrow="true" cta="View guide" />

  <Card title="PGVector" icon="link" href="/oss/python/integrations/vectorstores/pgvector" arrow="true" cta="View guide" />

  <Card title="PGVectorStore" icon="link" href="/oss/python/integrations/vectorstores/pgvectorstore" arrow="true" cta="View guide" />

  <Card title="Pinecone" icon="link" href="/oss/python/integrations/vectorstores/pinecone" arrow="true" cta="View guide" />

  <Card title="Pinecone (sparse)" icon="link" href="/oss/python/integrations/vectorstores/pinecone_sparse" arrow="true" cta="View guide" />

  <Card title="Qdrant" icon="link" href="/oss/python/integrations/vectorstores/qdrant" arrow="true" cta="View guide" />

  <Card title="Relyt" icon="link" href="/oss/python/integrations/vectorstores/relyt" arrow="true" cta="View guide" />

  <Card title="Rockset" icon="link" href="/oss/python/integrations/vectorstores/rockset" arrow="true" cta="View guide" />

  <Card title="SAP HANA Cloud Vector Engine" icon="link" href="/oss/python/integrations/vectorstores/sap_hanavector" arrow="true" cta="View guide" />

  <Card title="ScaNN" icon="link" href="/oss/python/integrations/vectorstores/scann" arrow="true" cta="View guide" />

  <Card title="SemaDB" icon="link" href="/oss/python/integrations/vectorstores/semadb" arrow="true" cta="View guide" />

  <Card title="SingleStore" icon="link" href="/oss/python/integrations/vectorstores/singlestore" arrow="true" cta="View guide" />

  <Card title="scikit-learn" icon="link" href="/oss/python/integrations/vectorstores/sklearn" arrow="true" cta="View guide" />

  <Card title="SQLiteVec" icon="link" href="/oss/python/integrations/vectorstores/sqlitevec" arrow="true" cta="View guide" />

  <Card title="SQLite-VSS" icon="link" href="/oss/python/integrations/vectorstores/sqlitevss" arrow="true" cta="View guide" />

  <Card title="SQLServer" icon="link" href="/oss/python/integrations/vectorstores/sqlserver" arrow="true" cta="View guide" />

  <Card title="StarRocks" icon="link" href="/oss/python/integrations/vectorstores/starrocks" arrow="true" cta="View guide" />

  <Card title="Supabase" icon="link" href="/oss/python/integrations/vectorstores/supabase" arrow="true" cta="View guide" />

  <Card title="SurrealDB" icon="link" href="/oss/python/integrations/vectorstores/surrealdb" arrow="true" cta="View guide" />

  <Card title="Tablestore" icon="link" href="/oss/python/integrations/vectorstores/tablestore" arrow="true" cta="View guide" />

  <Card title="Tair" icon="link" href="/oss/python/integrations/vectorstores/tair" arrow="true" cta="View guide" />

  <Card title="Tencent Cloud VectorDB" icon="link" href="/oss/python/integrations/vectorstores/tencentvectordb" arrow="true" cta="View guide" />

  <Card title="ThirdAI NeuralDB" icon="link" href="/oss/python/integrations/vectorstores/thirdai_neuraldb" arrow="true" cta="View guide" />

  <Card title="TiDB Vector" icon="link" href="/oss/python/integrations/vectorstores/tidb_vector" arrow="true" cta="View guide" />

  <Card title="Tigris" icon="link" href="/oss/python/integrations/vectorstores/tigris" arrow="true" cta="View guide" />

  <Card title="TileDB" icon="link" href="/oss/python/integrations/vectorstores/tiledb" arrow="true" cta="View guide" />

  <Card title="Timescale Vector" icon="link" href="/oss/python/integrations/vectorstores/timescalevector" arrow="true" cta="View guide" />

  <Card title="Typesense" icon="link" href="/oss/python/integrations/vectorstores/typesense" arrow="true" cta="View guide" />

  <Card title="Upstash Vector" icon="link" href="/oss/python/integrations/vectorstores/upstash" arrow="true" cta="View guide" />

  <Card title="USearch" icon="link" href="/oss/python/integrations/vectorstores/usearch" arrow="true" cta="View guide" />

  <Card title="Vald" icon="link" href="/oss/python/integrations/vectorstores/vald" arrow="true" cta="View guide" />

  <Card title="VDMS" icon="link" href="/oss/python/integrations/vectorstores/vdms" arrow="true" cta="View guide" />

  <Card title="Vearch" icon="link" href="/oss/python/integrations/vectorstores/vearch" arrow="true" cta="View guide" />

  <Card title="Vectara" icon="link" href="/oss/python/integrations/vectorstores/vectara" arrow="true" cta="View guide" />

  <Card title="Vespa" icon="link" href="/oss/python/integrations/vectorstores/vespa" arrow="true" cta="View guide" />

  <Card title="viking DB" icon="link" href="/oss/python/integrations/vectorstores/vikingdb" arrow="true" cta="View guide" />

  <Card title="vlite" icon="link" href="/oss/python/integrations/vectorstores/vlite" arrow="true" cta="View guide" />

  <Card title="Weaviate" icon="link" href="/oss/python/integrations/vectorstores/weaviate" arrow="true" cta="View guide" />

  <Card title="Xata" icon="link" href="/oss/python/integrations/vectorstores/xata" arrow="true" cta="View guide" />

  <Card title="YDB" icon="link" href="/oss/python/integrations/vectorstores/ydb" arrow="true" cta="View guide" />

  <Card title="Yellowbrick" icon="link" href="/oss/python/integrations/vectorstores/yellowbrick" arrow="true" cta="View guide" />

  <Card title="Zep" icon="link" href="/oss/python/integrations/vectorstores/zep" arrow="true" cta="View guide" />

  <Card title="Zep Cloud" icon="link" href="/oss/python/integrations/vectorstores/zep_cloud" arrow="true" cta="View guide" />

  <Card title="ZeusDB" icon="link" href="/oss/python/integrations/vectorstores/zeusdb" arrow="true" cta="View guide" />

  <Card title="Zilliz" icon="link" href="/oss/python/integrations/vectorstores/zilliz" arrow="true" cta="View guide" />
</Columns>

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/python/integrations/vectorstores/index.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.
</Tip>
