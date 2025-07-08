# wakili_engine.py

import os
import logging
import time
from dataclasses import dataclass, field
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pinecone import Pinecone
from typing import Dict, List, Tuple, Any
import re
import datetime
import pytz


# --- Enhanced Logging Setup ---
def setup_wakili_logging():
    """Configure comprehensive logging for the Wakili engine."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set specific log levels for external libraries
    logging.getLogger('google.generativeai').setLevel(logging.WARNING)
    logging.getLogger('pinecone').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return logger


logger = setup_wakili_logging()


# --- Configuration ---
@dataclass
class Config:
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv('PINECONE_API_KEY'))
    generative_model_name: str = 'gemini-2.0-flash'
    embedding_model_name: str = 'models/text-embedding-004'
    pinecone_index_name: str = "wakili-ai"
    namespace_mapping: Dict[str, str] = field(default_factory=lambda: {
        'constitution': 'constitution',
        'statutes': 'statute',
        'caselaw': 'caselaw'
    })
    constitution_relevance_threshold: float = 0.50
    statute_relevance_threshold: float = 0.40
    caselaw_relevance_threshold: float = 0.50
    max_constitution_in_context: int = 5
    max_statutes_in_context: int = 8
    max_caselaw_in_context: int = 3 # Reduced from 6 to 3

    def __post_init__(self):
        """Log configuration after initialization."""
        logger.info("Configuration loaded:")
        logger.info(f"  - Google API Key: {'✓ Set' if self.google_api_key else '✗ Missing'}")
        logger.info(f"  - Pinecone API Key: {'✓ Set' if self.pinecone_api_key else '✗ Missing'}")
        logger.info(f"  - Generative Model: {self.generative_model_name}")
        logger.info(f"  - Embedding Model: {self.embedding_model_name}")
        logger.info(f"  - Pinecone Index: {self.pinecone_index_name}")
        logger.info(f"  - Constitution Threshold: {self.constitution_relevance_threshold}")
        logger.info(f"  - Statute Threshold: {self.statute_relevance_threshold}")
        logger.info(f"  - Caselaw Threshold: {self.caselaw_relevance_threshold}")
        logger.info(f"  - Max Constitution Docs: {self.max_constitution_in_context}")
        logger.info(f"  - Max Statutes: {self.max_statutes_in_context}")
        logger.info(f"  - Max Caselaw: {self.max_caselaw_in_context}")


class WakiliAI:
    def __init__(self, config: Config):
        logger.info("Initializing WakiliAI instance...")
        self.config = config
        self.generative_model = None
        self.pinecone_index = None
        self.available_namespaces = []
        self._initialize_connections()
        logger.info("WakiliAI initialization complete")

    def _initialize_connections(self):
        """Initialize connections to Google AI and Pinecone with detailed logging."""
        logger.info("Starting connection initialization...")

        try:
            # Validate API keys
            if not all([self.config.google_api_key, self.config.pinecone_api_key]):
                missing_keys = []
                if not self.config.google_api_key:
                    missing_keys.append('GOOGLE_API_KEY')
                if not self.config.pinecone_api_key:
                    missing_keys.append('PINECONE_API_KEY')
                raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

            logger.info("API keys validated successfully")

            # Initialize Google AI
            logger.info("Configuring Google Generative AI...")
            genai.configure(api_key=self.config.google_api_key)
            self.generative_model = genai.GenerativeModel(self.config.generative_model_name)
            logger.info(f"Google AI model '{self.config.generative_model_name}' initialized")

            # Initialize Pinecone
            logger.info("Connecting to Pinecone...")
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.pinecone_index = pc.Index(self.config.pinecone_index_name)
            logger.info(f"Pinecone index '{self.config.pinecone_index_name}' connected")

            # Get namespace information
            logger.info("Retrieving index statistics...")
            stats = self.pinecone_index.describe_index_stats()
            self.available_namespaces = list(stats.get('namespaces', {}).keys())

            logger.info(f"Index statistics retrieved:")
            logger.info(f"  - Available namespaces: {self.available_namespaces}")

            # Log namespace details
            for namespace, info in stats.get('namespaces', {}).items():
                vector_count = info.get('vector_count', 0)
                logger.info(f"  - Namespace '{namespace}': {vector_count} vectors")

            logger.info("✅ All engine connections established successfully")

        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}", exc_info=True)
            raise

    def _extract_legal_keywords(self, question: str) -> Tuple[str, List[str], List[str]]:
        """Extract legal keywords from user question with detailed logging."""
        logger.info("Starting legal keyword extraction...")
        logger.debug(f"Input question: {question}")

        keyword_prompt = f"""You are an expert Kenyan paralegal. Your task is to analyze a user's question and extract key information for a legal database search.

From the user question below, extract the following:
1. PRIMARY LEGAL AREA: The specific area of Kenyan law.
2. KEY LEGAL TERMS: Important legal concepts, phrases, and synonyms.
3. SPECIFIC ACTIONS/ISSUES: The core actions or problems described.
4. RELEVANT ACTS: The specific Kenyan Acts that govern the issue. Be thorough. If the topic is about employment, include the Employment Act. If it's about a car accident, you MUST include the Traffic Act. For consumer rights, you MUST include the Consumer Protection Act and the Anti-Counterfeit Act.

**User Question to Analyze:** "{question}"
"""

        try:
            logger.debug("Sending keyword extraction prompt to AI...")
            start_time = time.time()

            response = self.generative_model.generate_content(keyword_prompt)

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(f"Keyword extraction completed in {processing_time:.2f} seconds")

            text = response.text.strip()
            logger.debug(f"Raw AI response length: {len(text)} characters")
            logger.info(f"Extracted Keywords Response:\n{text}")

            # --- FIX: Robust parsing for keyword extraction ---
            data = {'AREA': "", 'TERMS': [], 'ISSUES': [], 'ACTS': []}
            
            # Regex to find the content block for each section
            area_match = re.search(r"1\.\s*\*\*PRIMARY LEGAL AREA:\*\*(.*?)(?=\n2\.|\Z)", text, re.DOTALL | re.IGNORECASE)
            terms_match = re.search(r"2\.\s*\*\*KEY LEGAL TERMS:\*\*(.*?)(?=\n3\.|\Z)", text, re.DOTALL | re.IGNORECASE)
            issues_match = re.search(r"3\.\s*\*\*SPECIFIC ACTIONS/ISSUES:\*\*(.*?)(?=\n4\.|\Z)", text, re.DOTALL | re.IGNORECASE)
            acts_match = re.search(r"4\.\s*\*\*RELEVANT ACTS:\*\*(.*)", text, re.DOTALL | re.IGNORECASE)

            if area_match:
                data['AREA'] = area_match.group(1).strip()

            def extract_list_items(match_obj):
                if not match_obj:
                    return []
                content_block = match_obj.group(1)
                # Split by lines, strip whitespace and list markers (*, -, or numbers)
                items = [re.sub(r'^\s*[\*\-]\s*|\s*\d+\.\s*', '', line).strip() for line in content_block.split('\n') if line.strip()]
                # Further clean up by removing the bolded Act names if present
                cleaned_items = []
                for item in items:
                    cleaned_item = re.sub(r'\*\*(.*?):\*\*\s*', '', item)
                    if cleaned_item:
                        cleaned_items.append(cleaned_item.strip())
                return cleaned_items

            data['TERMS'] = extract_list_items(terms_match)
            data['ISSUES'] = extract_list_items(issues_match)
            data['ACTS'] = extract_list_items(acts_match)

            # Combine search terms
            search_terms = data['TERMS'] + data['ACTS']

            logger.info(f"Keyword extraction results:")
            logger.info(f"  - Legal Area: {data['AREA']}")
            logger.info(f"  - Legal Terms: {data['TERMS']}")
            logger.info(f"  - Legal Issues: {data['ISSUES']}")
            logger.info(f"  - Relevant Acts: {data['ACTS']}")
            logger.info(f"  - Combined Search Terms: {len(search_terms)} terms")

            return data['AREA'], search_terms, data['ISSUES']

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}", exc_info=True)
            return "", [], []

    def _search_vector_db(self, queries: List[str], namespace_key: str, top_k: int = 10) -> Dict[str, Any]:
        """Search vector database with comprehensive logging."""
        actual_namespace = self.config.namespace_mapping.get(namespace_key)

        logger.info(f"Starting vector search in namespace '{namespace_key}' -> '{actual_namespace}'")
        logger.debug(f"Search queries: {queries}")
        logger.debug(f"Top K: {top_k}")

        if not actual_namespace or actual_namespace not in self.available_namespaces:
            logger.warning(f"Namespace '{actual_namespace}' for key '{namespace_key}' not found or not available")
            logger.warning(f"Available namespaces: {self.available_namespaces}")
            return {}

        matches = {}

        try:
            # Generate embeddings
            logger.debug("Generating embeddings for search queries...")
            start_time = time.time()

            embeddings = genai.embed_content(
                model=self.config.embedding_model_name,
                content=queries
            )['embedding']

            embedding_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {embedding_time:.2f} seconds")

            # Get relevance threshold based on the namespace key
            if namespace_key == 'statutes':
                relevance_threshold = self.config.statute_relevance_threshold
            elif namespace_key == 'caselaw':
                relevance_threshold = self.config.caselaw_relevance_threshold
            elif namespace_key == 'constitution':
                relevance_threshold = self.config.constitution_relevance_threshold
            else:
                relevance_threshold = 0.4  # A sensible default

            logger.debug(f"Using relevance threshold: {relevance_threshold}")

            # Perform searches
            total_matches = 0
            for i, embedding in enumerate(embeddings):
                query_text = queries[i] if i < len(queries) else f"Query {i + 1}"
                logger.debug(f"Searching with query {i + 1}: '{query_text}'")

                search_start = time.time()
                results = self.pinecone_index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=actual_namespace
                )
                search_time = time.time() - search_start

                query_matches = results.get('matches', [])
                logger.debug(f"Query {i + 1} returned {len(query_matches)} results in {search_time:.2f}s")

                # Filter by relevance threshold
                relevant_matches = 0
                for match in query_matches:
                    score = match.get('score', 0)
                    if score >= relevance_threshold:
                        relevant_matches += 1
                        if match['id'] not in matches or matches[match['id']]['score'] < score:
                            matches[match['id']] = match
                            logger.debug(f"Added/updated match: {match['id']} (score: {score:.3f})")

                logger.debug(f"Query {i + 1}: {relevant_matches}/{len(query_matches)} matches above threshold")
                total_matches += relevant_matches

            logger.info(
                f"Vector search completed: {len(matches)} unique matches from {total_matches} total relevant results")

            if matches:
                # Log top matches
                sorted_matches = sorted(matches.values(), key=lambda x: x.get('score', 0), reverse=True)
                logger.info(f"Top 3 matches:")
                for i, match in enumerate(sorted_matches[:3]):
                    title = match.get('metadata', {}).get('title', 'N/A')
                    score = match.get('score', 0)
                    logger.info(f"  {i + 1}. {title} (score: {score:.3f})")

        except Exception as e:
            logger.error(f"Error during vector search in '{actual_namespace}': {e}", exc_info=True)

        return matches

    def _enhanced_search_strategy(self, question: str, legal_area: str, search_terms: List[str],
                                  legal_issues: List[str], namespace: str) -> Dict[str, Any]:
        """Enhanced search strategy with detailed logging."""
        logger.info(f"🔍 Starting enhanced search strategy for '{namespace}'")
        logger.debug(f"Legal area: {legal_area}")
        logger.debug(f"Search terms: {search_terms}")
        logger.debug(f"Legal issues: {legal_issues}")

        all_matches = {}

        # Build query list
        queries = [question]
        if legal_area and search_terms:
            combined_query = f"{legal_area} {' '.join(search_terms[:3])}"
            queries.append(combined_query)
            logger.debug(f"Added combined query: {combined_query}")

        queries.extend(search_terms[:5])
        queries.extend(legal_issues[:3])

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                unique_queries.append(q)
                seen.add(q)

        logger.info(f"Constructed {len(unique_queries)} unique search queries")

        if unique_queries:
            logger.debug("Executing primary search...")
            all_matches.update(self._search_vector_db(unique_queries, namespace, top_k=5))

        # Fallback search if no matches
        if not all_matches:
            logger.warning(f"No matches found for {namespace}. Trying simplified fallback search...")
            simple_query = " ".join(question.split()[:7])
            logger.debug(f"Fallback query: '{simple_query}'")
            all_matches.update(self._search_vector_db([simple_query], namespace, top_k=10))

            if all_matches:
                logger.info(f"Fallback search found {len(all_matches)} matches")
            else:
                logger.warning(f"Fallback search also returned no matches for {namespace}")

        logger.info(f"✅ Enhanced search for '{namespace}' completed: {len(all_matches)} unique matches")
        return all_matches

    def _format_context_section(self, matches: Dict[str, Any], doc_type: str, max_docs: int) -> Tuple[str, List[Dict]]:
        """Format context section with logging."""
        logger.debug(f"Formatting context section for {doc_type}")
        logger.debug(f"Input matches: {len(matches)}, Max docs: {max_docs}")

        sorted_matches = sorted(matches.values(), key=lambda x: x.get('score', 0), reverse=True)
        top_matches = sorted_matches[:max_docs]

        logger.debug(f"Selected top {len(top_matches)} matches for {doc_type}")

        context_str = ""
        metadata_list = []

        for i, match in enumerate(top_matches):
            metadata = match.get('metadata', {})
            title = metadata.get('title', 'N/A')
            text_snippet = metadata.get('text_snippet', '')
            score = match.get('score', 0)

            logger.debug(f"{doc_type} {i + 1}: {title} (score: {score:.3f}, length: {len(text_snippet)})")

            context_str += f"Source Type: {doc_type}\nTitle: {title}\nContent: {text_snippet}\n---\n"
            metadata_list.append(metadata)

        logger.info(
            f"Context section formatted: {len(top_matches)} {doc_type} documents, {len(context_str)} characters")
        return context_str, metadata_list

    def _build_context(self, constitution_matches: Dict, statute_matches: Dict, caselaw_matches: Dict) -> Tuple[str, List[Dict]]:
        """Build context from constitution, statute, and caselaw matches with detailed logging."""
        logger.info("Building context from constitution, statute and caselaw matches...")

        constitution_context, constitution_meta = self._format_context_section(
            constitution_matches, "Constitution", self.config.max_constitution_in_context
        )

        statute_context, statute_meta = self._format_context_section(
            statute_matches, "Statute", self.config.max_statutes_in_context
        )

        caselaw_context, caselaw_meta = self._format_context_section(
            caselaw_matches, "Case Law", self.config.max_caselaw_in_context
        )

        context = constitution_context + statute_context + caselaw_context
        source_metadata = constitution_meta + statute_meta + caselaw_meta

        logger.info(f"Context built successfully:")
        logger.info(f"  - Constitution: {len(constitution_meta)} documents")
        logger.info(f"  - Statutes: {len(statute_meta)} documents")
        logger.info(f"  - Case Law: {len(caselaw_meta)} documents")
        logger.info(f"  - Total context length: {len(context)} characters")
        logger.info(f"  - Total source metadata: {len(source_metadata)} entries")

        return context, source_metadata

    def _get_greeting(self):
        """Returns a time-appropriate greeting."""
        try:
            nairobi_tz = pytz.timezone("Africa/Nairobi")
            current_hour = datetime.datetime.now(nairobi_tz).hour
            if 5 <= current_hour < 12:
                return "Good morning!"
            elif 12 <= current_hour < 18:
                return "Good afternoon!"
            else:
                return "Good evening!"
        except Exception as e:
            logger.error(f"Could not determine time-based greeting: {e}")
            return "Hello!"


    def _generate_response(self, question: str, context: str) -> str:
        """Generate AI response with comprehensive logging."""
        logger.info("Starting AI response generation...")
        logger.debug(f"Question length: {len(question)} characters")
        logger.debug(f"Context length: {len(context)} characters")

        greeting = self._get_greeting()

        prompt = f"""You are Wakili Wangu, an expert legal AI assistant for Kenya. Your task is to provide a high-accuracy answer based ONLY on the provided legal context.

**Thinking Process (Chain of Thought):**
1.  **Analyze the User's Question:** Read the user's **Question** below to understand their specific problem.
2.  **Scan the Context for Key Laws:** Search the entire **Context** provided. Identify the primary Acts and case law that directly address the user's question.
3.  **Extract Specific Details:** Pull out the exact legal rules, rights, and penalties from the relevant documents.
4.  **Synthesize the Final Answer:** Based *only* on the extracted details, construct the final answer using the structure below.

**Response Structure (Use Telegram Markdown - *bold* and _italic_):**
- Start with a time-appropriate greeting ({greeting}), then in the same paragraph, provide a single, empathetic sentence showing you understand the user's situation. Do *not* use a title for this part.
- *✅ Direct Answer:* A clear, one-sentence summary of the legal position.
- *⚖️ The Law Explained:* Explain the most relevant Act or constitutional article from the context.
- *🏛️ Relevant Case Law:* If there are cases, summarize one that applies using a "•" bullet point. If not, state: "No specific case law was retrieved for this query."
- *📝 Recommended Steps:* A clear, numbered list of actions the user should take.

**Crucial Rule:** If the context does not contain the specific penalty or fine amount, you MUST state that clearly. Do not invent numbers.

**Context:**
---
{context}
---

**Question:** {question}

**Answer:**"""

        try:
            logger.debug("Configuring AI safety settings...")
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            logger.debug("Sending prompt to AI model...")
            start_time = time.time()

            response = self.generative_model.generate_content(prompt, safety_settings=safety_settings)

            end_time = time.time()
            generation_time = end_time - start_time

            logger.info(f"AI response generated successfully in {generation_time:.2f} seconds")

            response_text = response.text
            logger.info(f"Generated response length: {len(response_text)} characters")
            logger.debug(f"Response preview: {response_text[:200]}...")

            return response_text

        except Exception as e:
            logger.error(f"Error generating AI response: {e}", exc_info=True)
            return "I am sorry, I encountered an error while formulating the final response."

    def _format_sources_and_disclaimer(self, source_metadata: List[Dict]) -> str:
        """Format sources and disclaimer with logging."""
        logger.debug(f"Formatting sources from {len(source_metadata)} metadata entries")

        if not source_metadata:
            logger.debug("No source metadata provided")
            return ""

        # Extract unique sources
        unique_sources = {}
        for metadata in source_metadata:
            title = metadata.get('title', 'N/A').strip()
            url = metadata.get('source_url', '#')
            if title != 'N/A':
                unique_sources[title] = url

        logger.debug(f"Found {len(unique_sources)} unique sources")

        sources_list = []
        for title, url in unique_sources.items():
            # --- FIX: Check for valid URL vs. URN ---
            if url and url.startswith('http'):
                sources_list.append(f"- [{title}]({url})")
                logger.debug(f"Added source with URL: {title}")
            else:
                sources_list.append(f"- {title}")
                logger.debug(f"Added source without URL (or with URN): {title}")

        # --- FIX: Add version and date from environment variable ---
        data_last_updated = os.getenv('DATA_LAST_UPDATED', 'July 2025') # Default value for safety

        disclaimer = (
            "\n\n---\n"
            "_*Disclaimer:* This information is for educational purposes only and does not constitute legal advice. "
            "It is not a substitute for consultation with a qualified legal professional. "
            "Always consult an advocate for advice on your specific situation._"
            f"\n\n_Wakili Wangu V2.5 (Updated: {data_last_updated})_"
        )


        if not sources_list:
            logger.debug("No valid sources found, returning disclaimer only")
            return disclaimer

        formatted_sources = "\n\n" + "=" * 15 + "\n*Sources Used:*\n" + "\n".join(sources_list) + disclaimer
        logger.info(f"Sources section formatted with {len(sources_list)} sources")

        return formatted_sources

    def get_response(self, question: str) -> str:
        """Main method to get response with comprehensive logging."""
        logger.info("=" * 50)
        logger.info(f"🚀 NEW QUESTION PROCESSING STARTED")
        logger.info(f"Question: \"{question}\"")
        logger.info("=" * 50)

        # Check system status
        if not self.pinecone_index:
            logger.error("Pinecone index not available - system offline")
            return "My core systems are offline."

        overall_start_time = time.time()

        try:
            # Step 1: Extract legal keywords
            logger.info("📋 STEP 1: Extracting legal keywords...")
            legal_area, search_terms, legal_issues = self._extract_legal_keywords(question)

            # Step 2: Search the Constitution
            logger.info("📜 STEP 2: Searching the Constitution...")
            constitution_matches = self._enhanced_search_strategy(
                question, legal_area, search_terms, legal_issues, 'constitution'
            )

            # Step 3: Search for statutes
            logger.info("📚 STEP 3: Searching statute database...")
            statute_matches = self._enhanced_search_strategy(
                question, legal_area, search_terms, legal_issues, 'statutes'
            )

            # Step 4: Search for case law
            logger.info("⚖️ STEP 4: Searching case law database...")
            caselaw_matches = self._enhanced_search_strategy(
                question, legal_area, search_terms, legal_issues, 'caselaw'
            )

            # Step 5: Build context
            logger.info("🔨 STEP 5: Building legal context...")
            context, source_metadata = self._build_context(constitution_matches, statute_matches, caselaw_matches)

            # Check if we have any context
            if not context.strip():
                logger.warning("No relevant legal context found")
                return "I'm sorry, I could not find relevant legal information to answer your question."

            # Step 6: Generate AI response
            logger.info("🤖 STEP 6: Generating AI response...")
            answer = self._generate_response(question, context)

            # Step 7: Format sources
            logger.info("📄 STEP 7: Formatting sources and disclaimer...")
            sources_section = self._format_sources_and_disclaimer(source_metadata)

            # Combine final response
            final_response = answer + sources_section

            # Calculate total processing time
            total_time = time.time() - overall_start_time

            logger.info("✅ QUESTION PROCESSING COMPLETED SUCCESSFULLY")
            logger.info(f"📊 Processing Statistics:")
            logger.info(f"  - Total processing time: {total_time:.2f} seconds")
            logger.info(f"  - Constitution matches: {len(constitution_matches)}")
            logger.info(f"  - Statute matches: {len(statute_matches)}")
            logger.info(f"  - Case law matches: {len(caselaw_matches)}")
            logger.info(f"  - Context length: {len(context)} characters")
            logger.info(f"  - Final response length: {len(final_response)} characters")
            logger.info("=" * 50)

            return final_response

        except Exception as e:
            total_time = time.time() - overall_start_time
            logger.error(f"❌ CRITICAL ERROR in question processing after {total_time:.2f}s: {e}", exc_info=True)
            logger.error("=" * 50)
            return "I'm sorry, a critical error occurred while processing your question. Please try again."
