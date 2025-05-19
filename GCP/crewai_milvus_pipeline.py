# crewai_milvus_pipeline.py

import os
import openai
from crewai import Agent, Task, Crew
from pymilvus import connections, Collection

# ─── 0. CONFIGURATION ────────────────────────────────────────────────────────
# Zilliz Cloud (Milvus) credentials & collection
ZILLIZ_HOST       = "<YOUR_ZILLIZ_HOST>"
ZILLIZ_PORT       = "<YOUR_ZILLIZ_PORT>"       # e.g. "19530"
ZILLIZ_TOKEN      = "<YOUR_ZILLIZ_API_TOKEN>"
COLLECTION_NAME   = "<YOUR_COLLECTION_NAME>"

# OpenAI API key
OPENAI_API_KEY    = "<YOUR_OPENAI_API_KEY>"

# Embedding & retrieval settings
EMBED_MODEL       = "text-embedding-ada-002"
TOP_K             = 3
SEARCH_PARAMS     = {"metric_type": "L2", "params": {"nlist": 128}}

# LLM model for generation
LLM_MODEL         = "gpt-4.1-nano"


# ─── 1. SET UP CLIENTS ───────────────────────────────────────────────────────
# OpenAI
openai.api_key = OPENAI_API_KEY

# Milvus
connections.connect(
    alias="default",
    host=ZILLIZ_HOST,
    port=ZILLIZ_PORT,
    token=ZILLIZ_TOKEN
)
milvus_collection = Collection(COLLECTION_NAME)


# ─── 2. UTILITY FUNCTIONS ────────────────────────────────────────────────────
def embed_text(text: str):
    """Use OpenAI to embed text."""
    resp = openai.Embeddings.create(input=[text], model=EMBED_MODEL)
    return resp["data"][0]["embedding"]


def mcp_send(prompt: str) -> str:
    """MCP: send a prompt to the LLM and return its reply."""
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def a2a_send(sender: Agent, receiver: Agent, message: any):
    """A2A: pass a message from one agent to another."""
    return receiver.receive(message)


# ─── 3. DEFINE AGENTS ────────────────────────────────────────────────────────
# 3.1 Prompt Refiner Agent
prompt_refiner = Agent(
    role="Prompt Refiner",
    goal="Take a user query and generate 3 refined prompts to maximize retrieval quality.",
    backstory="I craft multiple query variations to help the retriever fetch the most relevant data.",
    tools=[mcp_send],
    allow_delegation=True,
    verbose=True
)

def prompt_refiner_receive(self, user_query: str):
    # Use query-expansion heuristics via LLM
    expansion_prompt = (
        "You are a clinical query engineer. Given the user question, generate exactly "
        "3 concise paraphrases or expansions that improve retrieval:\n\n"
        f"User Query: {user_query}\n\nVariants:"
    )
    variants = self.tools[0](expansion_prompt)
    # Expect variants separated by newlines
    return variants

prompt_refiner.receive = prompt_refiner_receive.__get__(prompt_refiner, Agent)


# 3.2 Retriever Agent
retriever = Agent(
    role="Retriever",
    goal="Retrieve top-K relevant records from the Milvus vector store for each query variant.",
    backstory="I fetch embeddings and query Milvus to find the best matching documents or images.",
    tools=[milvus_collection],
    allow_delegation=True,
    verbose=True
)

def retriever_receive(self, variants: str):
    hits = []
    for q in variants.splitlines():
        vec = embed_text(q)
        results = self.tools[0].search(
            data=[vec],
            anns_field="embeddings",
            param=SEARCH_PARAMS,
            limit=TOP_K,
            output_fields=["text", "image_id"]
        )
        hits.extend(results)
    return hits

retriever.receive = retriever_receive.__get__(retriever, Agent)


# 3.3 Summarization Agent
summarizer = Agent(
    role="Summarizer",
    goal="Synthesize retrieved snippets into a concise clinical answer.",
    backstory="I combine multiple source excerpts into a clear, accurate summary.",
    tools=[mcp_send],
    allow_delegation=False,
    verbose=True
)

def summarizer_receive(self, hits):
    # Combine text from retrieval hits
    combined = "\n\n".join(hit.entity.get("text", "") for hit in hits)
    summary_prompt = (
        "You are a clinical summarization expert. Given these excerpts, "
        "write a concise, accurate answer to the original question:\n\n"
        f"{combined}\n\nAnswer:"
    )
    return self.tools[0](summary_prompt)

summarizer.receive = summarizer_receive.__get__(summarizer, Agent)


# ─── 4. CREATE TASK & RUN CREW ───────────────────────────────────────────────
task = Task(
    description=(
        "User provides a clinical question. Generate refined queries, retrieve "
        "multimodal data from Milvus, and summarize into a final response."
    ),
    expected_output="A concise, clinically accurate answer.",
    agent=prompt_refiner
)

crew = Crew(
    agents=[prompt_refiner, retriever, summarizer],
    tasks=[task],
    verbose=True
)

if __name__ == "__main__":
    user_q = input("\nEnter clinical question: ")
    answer = crew.kickoff(user_q)
    print("\n=== Final Clinical Answer ===\n")
    print(answer)
