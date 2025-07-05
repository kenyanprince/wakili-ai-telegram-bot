import os
import logging
from dataclasses import dataclass, field
import google.generativeai as genai
from pinecone import Pinecone
from typing import Dict, List, Tuple, Any


# --- Configuration ---
@dataclass
class Config:
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv('PINECONE_API_KEY'))
    generative_model_name: str = 'gemini-1.5-flash'
    embedding_model_name: str = 'models/text-embedding-004'
    pinecone_index_name: str = "wakili-ai"


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WakiliAI:
    def __init__(self, config: Config):
        self.config = config
        self.generative_model = None
        self.pinecone_index = None
        self._initialize_connections()

    def _initialize_connections(self):
        try:
            if not all([self.config.google_api_key, self.config.pinecone_api_key]):
                raise ValueError("Missing GOOGLE_API_KEY or PINECONE_API_KEY")

            logger.info("Initializing connections to Google AI and Pinecone...")
            genai.configure(api_key=self.config.google_api_key)
            self.generative_model = genai.GenerativeModel(self.config.generative_model_name)
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.pinecone_index = pc.Index(self.config.pinecone_index_name)
            logger.info("✅ Engine connections successful.")
        except Exception as e:
            logger.error(f"❌ Engine failed to initialize: {e}", exc_info=True)
            raise

    def _get_context(self, question: str) -> Tuple[str, List[Dict]]:
        """Embeds question and queries Pinecone namespaces to retrieve context."""
        try:
            question_embedding = genai.embed_content(
                model=self.config.embedding_model_name,
                content=question
            )['embedding']

            # Query both namespaces
            statute_results = self.pinecone_index.query(
                vector=question_embedding, top_k=5, include_metadata=True, namespace='statutes'
            )
            caselaw_results = self.pinecone_index.query(
                vector=question_embedding, top_k=4, include_metadata=True, namespace='caselaw'
            )

            # Build context string and collect metadata
            context = ""
            source_metadata = []

            context_map = {
                "Statute": statute_results.get('matches', []),
                "Case Law": caselaw_results.get('matches', [])
            }

            for doc_type, matches in context_map.items():
                for match in matches:
                    metadata = match.get('metadata', {})
                    context += f"Source Type: {doc_type}\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
                    source_metadata.append(metadata)

            logger.info(
                f"Retrieved {len(statute_results.get('matches', []))} statutes and {len(caselaw_results.get('matches', []))} cases.")
            return context, source_metadata
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return "", []

    def _generate_response(self, question: str, context: str) -> str:
        """Generates the final user-facing response using the LLM and a detailed prompt."""
        prompt = f"""You are Wakili Wangu, an expert and empathetic legal AI assistant for Kenya. Your task is to answer the user's question with high accuracy, based ONLY on the provided legal context.

**Response Structure (use WhatsApp markdown):**
1.  *Empathetic Acknowledgment:* Start with a single sentence that shows you understand the user's situation.
2.  ✅ *Direct Answer:* Provide a clear, one-sentence summary of the legal position.
3.  ⚖️ *The Law Explained:*
    - Find the most relevant statute in the context.
    - Quote the specific section number and the law's text directly (e.g., "Section 49(1) of the Employment Act states...").
    - Explain what this law means in simple, practical terms.
4.  🏛️ *Relevant Case Law:*
    - If there are court cases in the context, summarize one that clearly illustrates how the law is applied in a real-world scenario.
    - If no relevant case law is in the context, state: "No specific case law was retrieved for this query."
5.  📝 *Recommended Steps:* Provide a clear, numbered list of actions the user should consider.

**Crucial Rules:**
- Do not invent information. If the context is insufficient, state: "Based on the information I have, I cannot provide a definitive answer. It is best to consult with a qualified advocate."
- Be professional, clear, and reassuring.

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**"""
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            return "I am sorry, I encountered an error while formulating the final response."

    def _format_sources_and_disclaimer(self, source_metadata: List[Dict]) -> str:
        """Creates the final sources and disclaimer string."""
        if not source_metadata: return ""

        unique_sources = {metadata.get('title', 'N/A').strip(): metadata.get('source_url', '#') for metadata in
                          source_metadata}
        sources_list = [f"- {title}" for title, url in unique_sources.items() if title != 'N/A']

        disclaimer = "\n\n---\n_Disclaimer: This is not legal advice. For informational purposes only. Always consult a qualified advocate._"
        if not sources_list: return disclaimer

        return "\n\n" + "=" * 15 + "\n*Sources Used:*\n" + "\n".join(sources_list) + disclaimer

    def get_response(self, question: str) -> str:
        """Main public method to get a response for a user question."""
        if not self.pinecone_index: return "My core systems are offline. Please try again later."
        try:
            logger.info(f"--- New Question Received: \"{question}\" ---")
            context, source_metadata = self._get_context(question)
            if not context.strip():
                return "I'm sorry, I could not find relevant legal information for your query. Please try rephrasing your question."

            answer = self._generate_response(question, context)
            sources_section = self._format_sources_and_disclaimer(source_metadata)
            return answer + sources_section
        except Exception as e:
            logger.error(f"A critical error occurred in get_response: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred. Please try again later."