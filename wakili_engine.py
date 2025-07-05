# wakili_engine.py

import os
import logging
from dataclasses import dataclass, field
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pinecone import Pinecone
from typing import Dict, List, Tuple, Any


# --- Configuration ---
@dataclass
class Config:
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv('PINECONE_API_KEY'))
    # IMPORTANT: Update this to your current model
    generative_model_name: str = 'gemini-1.5-flash'
    embedding_model_name: str = 'models/text-embedding-004'
    pinecone_index_name: str = "wakili-ai"
    namespace_mapping: Dict[str, str] = field(default_factory=lambda: {'statutes': 'statute', 'caselaw': 'caselaw'})
    statute_relevance_threshold: float = 0.40
    caselaw_relevance_threshold: float = 0.50
    max_statutes_in_context: int = 5  # Reduced for more focus
    max_caselaw_in_context: int = 3  # Reduced for more focus


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WakiliAI:
    def __init__(self, config: Config):
        self.config = config
        self.generative_model = None
        self.pinecone_index = None
        self.available_namespaces = []
        self._initialize_connections()

    def _initialize_connections(self):
        try:
            if not all([self.config.google_api_key, self.config.pinecone_api_key]):
                raise ValueError("Missing GOOGLE_API_KEY or PINECONE_API_KEY")
            logger.info("Initializing connections...")
            genai.configure(api_key=self.config.google_api_key)
            self.generative_model = genai.GenerativeModel(self.config.generative_model_name)
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.pinecone_index = pc.Index(self.config.pinecone_index_name)
            stats = self.pinecone_index.describe_index_stats()
            self.available_namespaces = list(stats.get('namespaces', {}).keys())
            logger.info("✅ Engine connections successful.")
        except Exception as e:
            logger.error(f"❌ Engine failed to initialize: {e}", exc_info=True)
            raise

    # --- ADVANCED RAG - STEP 1: Generate a Focused Search Query ---
    def _generate_search_query(self, question: str) -> str:
        """Uses the LLM to refine the user's question into a high-quality search query."""
        prompt = f"""You are a legal research expert. Your task is to convert a user's question into a concise, high-quality search query for a legal vector database. The query should be a short paragraph that summarizes the core legal issue.

User Question: "{question}"

Expert Search Query:"""
        try:
            response = self.generative_model.generate_content(prompt)
            logger.info(f"Generated expert search query: {response.text.strip()}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating search query: {e}. Falling back to original question.")
            return question

    def _search_vector_db(self, query: str) -> Tuple[str, List[Dict]]:
        """Performs a search on Pinecone using a single, high-quality query."""
        context, source_metadata = "", []
        try:
            query_embedding = genai.embed_content(
                model=self.config.embedding_model_name,
                content=query
            )['embedding']

            statute_results = self.pinecone_index.query(
                vector=query_embedding, top_k=self.config.max_statutes_in_context, include_metadata=True,
                namespace='statutes'
            )
            caselaw_results = self.pinecone_index.query(
                vector=query_embedding, top_k=self.config.max_caselaw_in_context, include_metadata=True,
                namespace='caselaw'
            )

            context_map = {"Statute": statute_results.get('matches', []),
                           "Case Law": caselaw_results.get('matches', [])}
            for doc_type, matches in context_map.items():
                for match in matches:
                    if match.get('score', 0) > self.config.statute_relevance_threshold:
                        metadata = match.get('metadata', {})
                        context += f"Source Type: {doc_type}\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
                        source_metadata.append(metadata)

            logger.info(
                f"Retrieved {len(statute_results.get('matches', []))} statutes and {len(caselaw_results.get('matches', []))} cases.")
        except Exception as e:
            logger.error(f"Error during vector search: {e}", exc_info=True)
        return context, source_metadata

    # --- ADVANCED RAG - STEP 2: Chain-of-Thought Synthesis ---
    def _generate_response(self, question: str, context: str) -> str:
        """Uses a Chain-of-Thought style prompt to force better reasoning."""
        prompt = f"""You are Wakili Wangu, an expert and empathetic legal AI assistant for Kenya. Your task is to provide a high-accuracy answer based ONLY on the provided legal context.

**Thinking Process (Chain of Thought):**
1.  **Identify the User's Core Problem:** What is the user's fundamental legal question?
2.  **Scan the Context for Key Laws:** Read through all the provided text. Identify the primary Acts that address the user's problem, such as the Consumer Protection Act or the Anti-Counterfeit Act.
3.  **Extract Specific Rules and Rights:** Pull out the exact duties of sellers and the rights of consumers mentioned in those Acts. Note any specific remedies like refunds or replacements.
4.  **Synthesize the Answer:** Based on the extracted rules, construct the final answer using the structure below. Do not add any outside information.

**Response Structure (use WhatsApp markdown):**
*   *Empathetic Acknowledgment:* Start with a single sentence showing you understand the situation.
*   ✅ *Direct Answer:* A clear, one-sentence summary of the user's rights.
*   ⚖️ *The Law Explained:* Explain the most relevant Act from the context (e.g., "The Consumer Protection Act..."). Quote specific sections if available. Explain what the law means for the user.
*   🏛️ *Relevant Case Law:* If there are cases, summarize one that applies. If not, state: "No specific case law was retrieved for this query."
*   📝 *Recommended Steps:* A clear, numbered list of actions the user should take based on the law.

**Crucial Rule:** If the context is insufficient to provide a detailed answer, state that clearly.

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**"""
        try:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            response = self.generative_model.generate_content(prompt, safety_settings=safety_settings)
            return response.text
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return "I am sorry, I encountered an error while formulating the final response."

    def _format_sources_and_disclaimer(self, source_metadata: List[Dict]) -> str:
        """Creates the final sources and disclaimer string with clickable links."""
        if not source_metadata: return ""
        unique_sources = {metadata.get('title', 'N/A').strip(): metadata.get('source_url', '#') for metadata in
                          source_metadata}
        sources_list = []
        for title, url in unique_sources.items():
            if title != 'N/A':
                if url and url != '#':
                    sources_list.append(f"- [{title}]({url})")
                else:
                    sources_list.append(f"- {title}")
        disclaimer = "\n\n---\n_Disclaimer: This is not legal advice..._"
        if not sources_list: return disclaimer
        return "\n\n" + "=" * 15 + "\n*Sources Used:*\n" + "\n".join(sources_list) + disclaimer

    # --- Main public method with the new, smarter flow ---
    def get_response(self, question: str) -> str:
        """Main public method to get a response for a user question."""
        if not self.pinecone_index: return "My core systems are offline."
        try:
            logger.info(f"--- New Question Received: \"{question}\" ---")

            # 1. Generate a high-quality search query
            expert_query = self._generate_search_query(question)

            # 2. Use the expert query to search the database
            context, source_metadata = self._search_vector_db(expert_query)

            if not context.strip():
                return "I'm sorry, I could not find relevant legal information for your query."

            # 3. Generate the final response using the Chain-of-Thought prompt
            answer = self._generate_response(question, context)

            sources_section = self._format_sources_and_disclaimer(source_metadata)
            return answer + sources_section
        except Exception as e:
            logger.error(f"A critical error occurred in get_response: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred."