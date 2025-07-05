# wakili_engine.py

import os
import logging
from dataclasses import dataclass, field
import google.generativeai as genai
from pinecone import Pinecone
from typing import Dict, List, Tuple, Any


# --- Configuration ---
@dataclass
class Config:
    """Centralized configuration for the AI assistant."""
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv('PINECONE_API_KEY'))
    generative_model_name: str = 'gemini-1.5-flash'
    embedding_model_name: str = 'models/text-embedding-004'
    pinecone_index_name: str = "wakili-ai"
    namespace_mapping: Dict[str, str] = field(default_factory=lambda: {'statutes': 'statute', 'caselaw': 'caselaw'})
    statute_relevance_threshold: float = 0.40
    caselaw_relevance_threshold: float = 0.50
    max_statutes_in_context: int = 8
    max_caselaw_in_context: int = 6

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WakiliAI:
    """A sophisticated legal AI assistant using a Retrieval-Augmented Generation (RAG) pipeline."""
    def __init__(self, config: Config):
        self.config = config
        self.generative_model = None
        self.pinecone_index = None
        self.available_namespaces = []
        self._initialize_connections()

    def _initialize_connections(self):
        try:
            if not all([self.config.google_api_key, self.config.pinecone_api_key]):
                raise ValueError("Missing required environment variables: GOOGLE_API_KEY, PINECONE_API_KEY")

            logger.info("Initializing connections to Google AI and Pinecone...")
            genai.configure(api_key=self.config.google_api_key)
            self.generative_model = genai.GenerativeModel(self.config.generative_model_name)

            pc = Pinecone(api_key=self.config.pinecone_api_key)
            if self.config.pinecone_index_name not in pc.list_indexes().names():
                raise NameError(f"Index '{self.config.pinecone_index_name}' does not exist.")
            self.pinecone_index = pc.Index(self.config.pinecone_index_name)
            stats = self.pinecone_index.describe_index_stats()
            self.available_namespaces = list(stats.get('namespaces', {}).keys())
            logger.info(f"Available namespaces cached: {self.available_namespaces}")
            logger.info("✅ Engine connections successful.")
        except Exception as e:
            logger.error(f"❌ Engine failed to initialize: {e}", exc_info=True)
            raise

    def _extract_legal_keywords(self, question: str) -> Tuple[str, List[str], List[str]]:
        keyword_prompt = f"""You are an expert Kenyan paralegal. Your task is to analyze a user's question and extract key information for a legal database search.

From the user question below, extract the following:
1. PRIMARY LEGAL AREA: The specific area of Kenyan law.
2. KEY LEGAL TERMS: Important legal concepts, phrases, and synonyms.
3. SPECIFIC ACTIONS/ISSUES: The core actions or problems described.
4. RELEVANT ACTS: The specific Kenyan Acts that govern the issue. Be thorough. If the topic is about employment, include the Employment Act. If it's about a car accident, you MUST include the Traffic Act. If it's about inheritance, include the Law of Succession Act.

**Example:**
User Question: "My boss fired me without giving me a letter."
Your Output:
AREA: Employment Law
TERMS: wrongful dismissal, unfair termination, termination notice, summary dismissal
ISSUES: termination without notice, procedural fairness, dismissal letter
ACTS: Employment Act, 2007

**User Question to Analyze:** "{question}"
"""
        try:
            response = self.generative_model.generate_content(keyword_prompt)
            text = response.text.strip()
            logger.info(f"Extracted Keywords:\n{text}")
            data = {'AREA': "", 'TERMS': [], 'ISSUES': [], 'ACTS': []}
            for line in text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    if key in data:
                        if isinstance(data[key], list):
                            data[key] = [item.strip() for item in value.strip().split(',') if item.strip()]
                        else:
                            data[key] = value.strip()
            search_terms = data['TERMS'] + data['ACTS']
            return data['AREA'], search_terms, data['ISSUES']
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}", exc_info=True)
            return "", [], []

    def _search_vector_db(self, queries: List[str], namespace_key: str, top_k: int = 10) -> Dict[str, Any]:
        actual_namespace = self.config.namespace_mapping.get(namespace_key)
        if not actual_namespace or actual_namespace not in self.available_namespaces:
            return {}
        matches = {}
        try:
            embeddings = genai.embed_content(model=self.config.embedding_model_name, content=queries)['embedding']
            relevance_threshold = self.config.statute_relevance_threshold if namespace_key == 'statutes' else self.config.caselaw_relevance_threshold
            for i, embedding in enumerate(embeddings):
                results = self.pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace=actual_namespace)
                for match in results.get('matches', []):
                    if match.get('score', 0) >= relevance_threshold:
                        if match['id'] not in matches or matches[match['id']]['score'] < match['score']:
                            matches[match['id']] = match
        except Exception as e:
            logger.error(f"Error during vector search in '{actual_namespace}': {e}", exc_info=True)
        return matches

    def _enhanced_search_strategy(self, question: str, legal_area: str, search_terms: List[str], legal_issues: List[str], namespace: str) -> Dict[str, Any]:
        logger.info(f"--- Running search strategy for '{namespace}' ---")
        all_matches = {}
        queries = [question]
        if legal_area and search_terms:
            queries.append(f"{legal_area} {' '.join(search_terms[:3])}")
        queries.extend(search_terms[:5])
        queries.extend(legal_issues[:3])
        unique_queries = sorted(list(set(q for q in queries if q)))
        if unique_queries:
            all_matches.update(self._search_vector_db(unique_queries, namespace, top_k=5))
        if not all_matches:
            logger.warning(f"No matches found for {namespace}. Trying simplified fallback.")
            simple_query = " ".join(question.split()[:7])
            all_matches.update(self._search_vector_db([simple_query], namespace, top_k=10))
        logger.info(f"--- Total unique matches for '{namespace}': {len(all_matches)} ---")
        return all_matches

    def _format_context_section(self, matches: Dict[str, Any], doc_type: str, max_docs: int) -> Tuple[str, List[Dict]]:
        sorted_matches = sorted(matches.values(), key=lambda x: x.get('score', 0), reverse=True)
        top_matches = sorted_matches[:max_docs]
        context_str = ""
        metadata_list = []
        for match in top_matches:
            metadata = match.get('metadata', {})
            context_str += f"Source Type: {doc_type}\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
            metadata_list.append(metadata)
        return context_str, metadata_list

    def _build_context(self, statute_matches: Dict, caselaw_matches: Dict) -> Tuple[str, List[Dict]]:
        statute_context, statute_meta = self._format_context_section(statute_matches, "Statute", self.config.max_statutes_in_context)
        caselaw_context, caselaw_meta = self._format_context_section(caselaw_matches, "Case Law", self.config.max_caselaw_in_context)
        context = statute_context + caselaw_context
        source_metadata = statute_meta + caselaw_meta
        logger.info(f"Building context with {len(statute_meta)} statutes and {len(caselaw_meta)} cases. Total length: {len(context)} chars.")
        return context, source_metadata

    def _generate_legal_response(self, question: str, context: str, has_statutes: bool, has_caselaw: bool) -> str:
        """Generates the final user-facing response using the LLM."""
        context_guidance = ""
        if has_statutes and has_caselaw:
            context_guidance = "Prioritize statutes but use case law for practical application."
        elif has_statutes:
            context_guidance = "Focus on the provided statutory sections."
        elif has_caselaw:
            context_guidance = "Provide guidance based on the judicial decisions."
        else:
            context_guidance = "The context is limited. Provide general guidance and strongly recommend consulting a legal professional."

        # --- THIS IS THE ENHANCED PROMPT ---
        prompt = f"""You are Wakili Wangu, an expert legal AI assistant for Kenya.
Your task is to answer the user's question with high accuracy, based *ONLY* on the provided legal context.

**Context Guidance:** {context_guidance} Scour the context for specific laws and section numbers. If the context mentions the Traffic Act or Penal Code, you MUST prioritize explaining the sections related to the user's question.

**Response Structure:**
1. ✅ *Direct Answer:* A clear, one-sentence summary of the legal position.
2. ⚖️ *The Law Explained:*
   - Find the most relevant statute in the context (e.g., The Traffic Act).
   - Quote the specific section number and the law's text directly (e.g., "Section 49(1) of the Traffic Act states...").
   - Explain what this law means in simple, practical terms for the user.
3. 🏛️ *Relevant Case Law:* If there are court cases in the context, summarize one that illustrates how the law is applied in a real-world scenario.
4. 📝 *Recommended Steps:* Provide a clear, numbered list of actions the user should take based on the law.

**Important:**
- Do not invent information. If the context doesn't have the answer, say so.
- Use WhatsApp markdown (*bold*).
- Be compassionate, professional, and clear.

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**"""
        # --- END OF PROMPT ENHANCEMENT ---
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I'm sorry, an error occurred while generating the final response."

    def _format_sources(self, source_metadata: List[Dict]) -> str:
        """Creates a formatted string of unique sources and a disclaimer."""
        if not source_metadata:
            return ""
        unique_sources = {metadata.get('title', 'N/A').strip(): metadata.get('source_url', '#') for metadata in source_metadata}
        sources_list = [f"- {title} (URL: {url})" for title, url in unique_sources.items() if title != 'N/A']
        disclaimer = "\n\n---\n_Disclaimer: This is not legal advice. For informational purposes only. Always consult a qualified advocate._"
        if not sources_list:
            return disclaimer
        return "\n\n" + "=" * 15 + "\n*Sources Used:*\n" + "\n".join(sources_list) + disclaimer

    def get_wakili_response(self, question: str) -> str:
        """The main public method that orchestrates the entire RAG pipeline."""
        if not self.generative_model or not self.pinecone_index:
            return "I'm sorry, my core systems are not online. Please check the logs."
        try:
            logger.info(f"--- New Question Received: \"{question}\" ---")
            legal_area, search_terms, legal_issues = self._extract_legal_keywords(question)
            statute_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues, 'statutes')
            caselaw_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues, 'caselaw')
            context, source_metadata = self._build_context(statute_matches, caselaw_matches)
            if not context.strip():
                return "I'm sorry, I could not find relevant legal information for your query. Please try rephrasing or consult a legal professional."
            answer = self._generate_legal_response(question, context, bool(statute_matches), bool(caselaw_matches))
            sources_section = self._format_sources(source_metadata)
            return answer + sources_section
        except Exception as e:
            logger.error(f"A critical error occurred in get_wakili_response: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred. Please try again."