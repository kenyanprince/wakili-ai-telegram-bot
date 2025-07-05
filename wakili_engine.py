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
4. RELEVANT ACTS: The specific Kenyan Acts that govern the issue. Be thorough. If the topic is about employment, include the Employment Act. If it's about a car accident, you MUST include the Traffic Act. For consumer rights, you MUST include the Consumer Protection Act and the Anti-Counterfeit Act.

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
                    key = key.strip().upper().replace('*', '')
                    if key.startswith("1. PRIMARY LEGAL AREA"):
                        key = "AREA"
                    elif key.startswith("2. KEY LEGAL TERMS"):
                        key = "TERMS"
                    elif key.startswith("3. SPECIFIC ACTIONS/ISSUES"):
                        key = "ISSUES"
                    elif key.startswith("4. RELEVANT ACTS"):
                        key = "ACTS"

                    if key in data:
                        clean_value = value.replace('*', '').strip()
                        if isinstance(data[key], list):
                            data[key].extend([item.strip() for item in clean_value.split(',') if item.strip()])
                        else:
                            data[key] = clean_value

            search_terms = data['TERMS'] + data['ACTS']
            return data['AREA'], search_terms, data['ISSUES']
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}", exc_info=True)
            return "", [], []

    def _search_vector_db(self, queries: List[str], namespace_key: str, top_k: int = 10) -> Dict[str, Any]:
        actual_namespace = self.config.namespace_mapping.get(namespace_key)
        if not actual_namespace or actual_namespace not in self.available_namespaces:
            logger.warning(f"Namespace '{actual_namespace}' for key '{namespace_key}' not found or not available.")
            return {}
        matches = {}
        try:
            embeddings = genai.embed_content(model=self.config.embedding_model_name, content=queries)['embedding']
            relevance_threshold = self.config.statute_relevance_threshold if namespace_key == 'statutes' else self.config.caselaw_relevance_threshold
            for i, embedding in enumerate(embeddings):
                results = self.pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True,
                                                    namespace=actual_namespace)
                for match in results.get('matches', []):
                    if match.get('score', 0) >= relevance_threshold:
                        if match['id'] not in matches or matches[match['id']]['score'] < match['score']:
                            matches[match['id']] = match
        except Exception as e:
            logger.error(f"Error during vector search in '{actual_namespace}': {e}", exc_info=True)
        return matches

    def _enhanced_search_strategy(self, question: str, legal_area: str, search_terms: List[str],
                                  legal_issues: List[str], namespace: str) -> Dict[str, Any]:
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
        context_str, metadata_list = "", []
        for match in top_matches:
            metadata = match.get('metadata', {})
            context_str += f"Source Type: {doc_type}\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
            metadata_list.append(metadata)
        return context_str, metadata_list

    def _build_context(self, statute_matches: Dict, caselaw_matches: Dict) -> Tuple[str, List[Dict]]:
        statute_context, statute_meta = self._format_context_section(statute_matches, "Statute",
                                                                     self.config.max_statutes_in_context)
        caselaw_context, caselaw_meta = self._format_context_section(caselaw_matches, "Case Law",
                                                                     self.config.max_caselaw_in_context)
        context = statute_context + caselaw_context
        source_metadata = statute_meta + caselaw_meta
        logger.info(f"Building context with {len(statute_meta)} statutes and {len(caselaw_meta)} cases.")
        return context, source_metadata

    def _generate_response(self, question: str, context: str) -> str:
        # --- THIS IS THE CORRECTED, DYNAMIC PROMPT ---
        prompt = f"""You are Wakili Wangu, an expert legal AI assistant for Kenya. Your task is to provide a high-accuracy answer based ONLY on the provided legal context.

**Thinking Process (Chain of Thought):**
1.  **Analyze the User's Question:** Read the user's **Question** below to understand their specific problem.
2.  **Scan the Context for Relevant Laws:** Search the entire **Context** provided. Identify the primary Acts and case law that directly address the user's question.
3.  **Extract Specific Rules and Principles:** Pull out the exact legal rules, rights, duties, and penalties from the most relevant documents in the context.
4.  **Synthesize the Final Answer:** Based *only* on the extracted rules, construct the final answer using the structure below. Directly connect the legal principles to the user's situation.

**Response Structure (use WhatsApp markdown):**
*   *Empathetic Acknowledgment:* Start with a single sentence showing you understand the user's situation.
*   ✅ *Direct Answer:* A clear, one-sentence summary of the legal position based on the context.
*   ⚖️ *The Law Explained:* Explain the most relevant Act from the context (e.g., "The Traffic Act says..."). Quote specific sections if available. Explain what the law means for the user.
*   🏛️ *Relevant Case Law:* If there are cases in the context, summarize one that applies. If not, state: "No specific case law was retrieved for this query."
*   📝 *Recommended Steps:* A clear, numbered list of actions the user should take based on the provided law.

**Crucial Rule:** If the context is insufficient to provide a detailed answer, you MUST state that clearly. Do not invent information.

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

    def get_response(self, question: str) -> str:
        if not self.pinecone_index: return "My core systems are offline."
        try:
            logger.info(f"--- New Question Received: \"{question}\" ---")
            legal_area, search_terms, legal_issues = self._extract_legal_keywords(question)
            statute_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues,
                                                             'statutes')
            caselaw_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues,
                                                             'caselaw')
            context, source_metadata = self._build_context(statute_matches, caselaw_matches)
            if not context.strip():
                return "I'm sorry, I could not find relevant legal information."
            answer = self._generate_response(question, context)
            sources_section = self._format_sources_and_disclaimer(source_metadata)
            return answer + sources_section
        except Exception as e:
            logger.error(f"A critical error occurred in get_response: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred."