from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions, Modality, Content, Part
from google.genai.types import Tool, FunctionDeclaration, Schema
from google.genai.types import Retrieval, VertexAISearch, VertexRagStore, VertexRagStoreRagResource
from google.genai.types import (
    GenerateContentConfig,
    Retrieval,
    Tool,
    VertexRagStore,
    VertexRagStoreRagResource,
)
from vertexai import rag

RETRIEVAL_QUERY = "When was Wild Irishman established?"  # @param {type:"string"}

rag_resource = rag.RagResource(
    rag_corpus="projects/sandbox-ampelic/locations/us-central1/ragCorpora/6838716034162098176",
    # Need to manually get the ids from rag.list_files.
    # rag_file_ids=[],
)

response = rag.retrieval_query(
    rag_resources=[rag_resource],  # Currently only 1 corpus is allowed.
    text=RETRIEVAL_QUERY,
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=10,  # Optional
        filter=rag.Filter(
            vector_distance_threshold=0.5,  # Optional
        ),
    ),
)

# The retrieved context can be passed to any SDK or model generation API to generate final results.
retrieved_context = " ".join(
    [context.text for context in response.contexts.contexts]
).replace("\n", "")

print(retrieved_context)