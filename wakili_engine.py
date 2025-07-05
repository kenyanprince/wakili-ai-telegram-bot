import os
import logging
import google.generativeai as genai
from pinecone import Pinecone
from typing import Dict, List, Tuple

# Set up logging to display informational messages
# This helps in tracking the bot's operations in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WakiliAI:
    """
    A sophisticated legal AI assistant that uses a Retrieval-Augmented Generation (RAG)
    pipeline. It integrates Google's Generative AI for language understanding and
    response generation, and Pinecone's vector database for efficient retrieval of
    relevant legal documents (statutes and case law).
    """

    def __init__(self):
        """Initialize the Wakili AI legal assistant."""
        self.generative_model = None
        self.pinecone_index = None
        # The name of the Pinecone index where legal documents are stored.
        self.INDEX_NAME = "wakili-ai"
        self._initialize_connections()

    def _initialize_connections(self):
        """
        Establishes and validates connections to essential third-party services:
        Google AI (for the Gemini model) and Pinecone (for the vector database).
        It securely loads API keys from environment variables.
        """
        try:
            # Load API keys from environment variables for security
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

            # Ensure that both API keys are present
            if not all([GOOGLE_API_KEY, PINECONE_API_KEY]):
                raise ValueError("Missing required environment variables: GOOGLE_API_KEY, PINECONE_API_KEY")

            logger.info("Initializing connections to Google AI and Pinecone...")

            # Configure the Google Generative AI client
            genai.configure(api_key=GOOGLE_API_KEY)
            self.generative_model = genai.GenerativeModel('gemini-1.5-flash')

            # Initialize the Pinecone client
            pc = Pinecone(api_key=PINECONE_API_KEY)

            # Check if the specified index exists in the Pinecone project
            available_indexes = pc.list_indexes().names()
            if self.INDEX_NAME not in available_indexes:
                logger.error(f"Index '{self.INDEX_NAME}' not found. Available indexes: {available_indexes}")
                raise NameError(f"Index '{self.INDEX_NAME}' does not exist in your Pinecone project.")

            self.pinecone_index = pc.Index(self.INDEX_NAME)

            # Log the available namespaces within the index for debugging purposes
            stats = self.pinecone_index.describe_index_stats()
            available_namespaces = list(stats.get('namespaces', {}).keys())
            logger.info(f"Available namespaces: {available_namespaces}")

            logger.info("✅ Engine connections successful.")

        except Exception as e:
            logger.error(f"❌ Engine failed to initialize: {e}")
            self.generative_model = None
            self.pinecone_index = None
            raise

    def _extract_legal_keywords(self, question: str) -> Tuple[str, List[str], List[str]]:
        """
        Uses the generative model to analyze the user's question and extract
        structured legal keywords, which are crucial for a targeted vector search.
        """
        keyword_prompt = f"""From the following user question, extract relevant legal search terms for Kenya:

1. PRIMARY LEGAL AREA: The main area of law
2. KEY LEGAL TERMS: Important legal concepts and terms
3. SPECIFIC ACTIONS/ISSUES: Concrete actions, violations, or legal issues
4. RELEVANT ACTS: Kenyan laws that might apply (e.g., Employment Act, Companies Act, etc.)

Format your response as:
AREA: [legal area]
TERMS: [term1, term2, term3, term4, term5]
ISSUES: [issue1, issue2, issue3]
ACTS: [act1, act2, act3]

User Question: "{question}"
"""
        try:
            keyword_response = self.generative_model.generate_content(keyword_prompt)
            keyword_text = keyword_response.text.strip()
            logger.info(f"Extracted Keywords:\n{keyword_text}")

            # Parse the structured text response from the model
            lines = keyword_text.split('\n')
            legal_area = ""
            search_terms = []
            legal_issues = []
            relevant_acts = []

            for line in lines:
                line = line.strip()
                if line.startswith("AREA:"):
                    legal_area = line.replace("AREA:", "").strip()
                elif line.startswith("TERMS:"):
                    terms_text = line.replace("TERMS:", "").strip()
                    search_terms = [term.strip() for term in terms_text.split(',') if term.strip()]
                elif line.startswith("ISSUES:"):
                    issues_text = line.replace("ISSUES:", "").strip()
                    legal_issues = [issue.strip() for issue in issues_text.split(',') if issue.strip()]
                elif line.startswith("ACTS:"):
                    acts_text = line.replace("ACTS:", "").strip()
                    relevant_acts = [act.strip() for act in acts_text.split(',') if act.strip()]

            # Combine terms and acts for a comprehensive search list
            return legal_area, search_terms + relevant_acts, legal_issues

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return "", [], []

    def _search_vector_db(self, queries: List[str], namespace: str, top_k: int = 10) -> Dict:
        """
        Performs a batch search on the Pinecone vector database. It converts
        text queries into embeddings and retrieves the most similar documents.
        """
        matches = {}
        stats = self.pinecone_index.describe_index_stats()
        available_namespaces = list(stats.get('namespaces', {}).keys())

        # Map internal search logic to actual database namespaces
        namespace_mapping = {
            'statutes': 'statute',
            'caselaw': 'caselaw'
        }
        actual_namespace = namespace_mapping.get(namespace, namespace)

        if actual_namespace not in available_namespaces:
            logger.warning(f"Namespace '{actual_namespace}' not found. Available: {available_namespaces}")
            return matches

        logger.info(f"Using namespace '{actual_namespace}' for search type '{namespace}'")

        try:
            # Generate embeddings for all queries in a single API call for efficiency
            embeddings_response = genai.embed_content(
                model="models/text-embedding-004",
                content=queries
            )
            embeddings = embeddings_response['embedding']

            for i, embedding in enumerate(embeddings):
                query_text = queries[i]
                logger.info(f"Searching {actual_namespace} with query: '{query_text}'")

                try:
                    results = self.pinecone_index.query(
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=actual_namespace
                    )
                    logger.info(f"Search returned {len(results.get('matches', []))} results")

                    for match in results.get('matches', []):
                        score = match.get('score', 0)
                        # Use a dynamic relevance threshold based on content type
                        relevance_threshold = 0.4 if actual_namespace == 'statute' else 0.5

                        if score >= relevance_threshold:
                            match_id = match['id']
                            # Add to matches, replacing if a more relevant result for the same document is found
                            if match_id not in matches or matches[match_id]['score'] < score:
                                matches[match_id] = match
                                metadata = match.get('metadata', {})
                                title = metadata.get('title', 'N/A')
                                logger.info(f"  -> Found relevant match: {title} (score: {score:.3f})")

                except Exception as e:
                    logger.error(f"Error during Pinecone query for '{query_text}': {e}")

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")

        return matches

    def _enhanced_search_strategy(self, question: str, legal_area: str, search_terms: List[str],
                                  legal_issues: List[str], namespace: str) -> Dict:
        """
        Executes a multi-pronged search strategy to ensure comprehensive document
        retrieval, from broad queries to very specific term searches.
        """
        all_matches = {}

        # Strategy 1: Search using the user's question directly
        logger.info(f"--- Running search strategy for '{namespace}' ---")
        direct_matches = self._search_vector_db([question], namespace, top_k=5)
        all_matches.update(direct_matches)

        # Strategy 2: Search using a combination of the legal area and key terms
        if legal_area and search_terms:
            combined_query = f"{legal_area} {' '.join(search_terms[:3])}"
            area_matches = self._search_vector_db([combined_query], namespace, top_k=5)
            all_matches.update(area_matches)

        # Strategy 3: Search using individual key terms and issues
        if search_terms or legal_issues:
            individual_queries = (search_terms[:5] + legal_issues[:3])
            term_matches = self._search_vector_db(individual_queries, namespace, top_k=3)
            all_matches.update(term_matches)

        # Strategy 4: Fallback search if no matches are found initially
        if not all_matches:
            logger.warning(f"No matches found for {namespace}. Trying simplified fallback search...")
            simple_query = " ".join(question.split()[:7]) # Use first 7 words as a simple query
            fallback_matches = self._search_vector_db([simple_query], namespace, top_k=10)
            all_matches.update(fallback_matches)

        logger.info(f"--- Total unique matches found for '{namespace}': {len(all_matches)} ---")
        return all_matches

    def _build_context(self, statute_matches: Dict, caselaw_matches: Dict) -> Tuple[str, List[Dict]]:
        """
        Constructs the final context string to be sent to the LLM. It prioritizes
        the highest-scoring and most relevant documents.
        """
        # Sort all retrieved matches by their relevance score
        sorted_statutes = sorted(statute_matches.values(), key=lambda x: x.get('score', 0), reverse=True)
        sorted_caselaw = sorted(caselaw_matches.values(), key=lambda x: x.get('score', 0), reverse=True)

        # Limit the number of documents to keep the context focused and within model limits
        top_statutes = sorted_statutes[:8]
        top_caselaw = sorted_caselaw[:6]

        logger.info(f"Building context with {len(top_statutes)} statutes and {len(top_caselaw)} cases.")
        context = ""
        source_metadata = []

        # Format statute context
        for match in top_statutes:
            metadata = match.get('metadata', {})
            context += f"Source Type: Statute\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
            source_metadata.append(metadata)

        # Format case law context
        for match in top_caselaw:
            metadata = match.get('metadata', {})
            context += f"Source Type: Case Law\nTitle: {metadata.get('title', 'N/A')}\nContent: {metadata.get('text_snippet', '')}\n---\n"
            source_metadata.append(metadata)

        logger.info(f"Final context length: {len(context)} characters")
        return context, source_metadata

    def _generate_legal_response(self, question: str, context: str, has_statutes: bool, has_caselaw: bool) -> str:
        """
        Generates the final, user-facing response by sending the user's question
        and the retrieved context to the Gemini model with specific instructions.
        """
        context_guidance = ""
        if has_statutes and has_caselaw:
            context_guidance = "You have both statutory provisions and case law. Prioritize statutes but use case law for practical application."
        elif has_statutes:
            context_guidance = "You have statutory provisions. Focus on the relevant legal sections."
        elif has_caselaw:
            context_guidance = "You only have case law. Provide guidance based on judicial decisions."
        else:
            context_guidance = "Limited legal context is available. Provide general guidance and strongly recommend consulting a legal professional."

        # This is the detailed prompt that guides the LLM's response.
        # *** The key change is here, in the Response Structure section. ***
        prompt = f"""You are Wakili Wangu, a helpful legal assistant for Kenya. Answer based ONLY on the provided context.

**Context Guidance:** {context_guidance}

**Response Structure:**
*Introduction:* Start with a short, polite, and empathetic opening. Acknowledge the user's situation before diving into the legal points. For example: "I understand this is a stressful situation, let's look at what the law says." or "Thank you for your question. Here's a breakdown of the rules on..."
1. ✅ *Direct Answer:* One-sentence summary of the legal position.
2. ⚖️ *The Law Explained:* 
   - Quote specific statutory provisions with Act name and section numbers when available.
   - Explain how the law applies to the user's situation.
3. 🏛️ *Relevant Case Law:* Summarize key cases that illustrate the legal principles.
4. 📝 *Recommended Steps:* Practical, numbered steps the user can take.

**Important:**
- Use WhatsApp markdown (single asterisks for bold).
- Be compassionate and professional.
- Quote legal provisions directly when available.
- Acknowledge the limitations of your knowledge base.

**Context:**
{context}

**Question:** {question}

**Answer:**"""

        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating the final response."

    def _format_sources(self, source_metadata: List[Dict]) -> str:
        """
        Creates a formatted string of unique sources used in the response,
        including a disclaimer.
        """
        sources_header = "\n\n" + "=" * 15 + "\n*Sources Used:*\n"
        unique_titles = set()
        sources_list = []

        for metadata in source_metadata:
            title = metadata.get('title', 'N/A').strip()
            # Add to list only if the title is not already included
            if title and title not in unique_titles:
                url = metadata.get('source_url', '#')
                sources_list.append(f"- {title} (URL: {url})")
                unique_titles.add(title)

        disclaimer = "\n\n---\n_Disclaimer: This is not legal advice. For informational purposes only. Always consult a qualified advocate._"
        return sources_header + "\n".join(sources_list) + disclaimer

    def get_wakili_response(self, question: str) -> str:
        """
        The main public method that orchestrates the entire RAG pipeline from
        question to final formatted response.
        """
        if not self.generative_model or not self.pinecone_index:
            return "I'm sorry, my core systems are not online right now. Please check the system logs."

        try:
            logger.info(f"Processing question: \"{question}\"")

            # 1. Understand the query
            legal_area, search_terms, legal_issues = self._extract_legal_keywords(question)

            # 2. Retrieve relevant documents (statutes and case law)
            statute_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues, 'statutes')
            caselaw_matches = self._enhanced_search_strategy(question, legal_area, search_terms, legal_issues, 'caselaw')

            # 3. Build the context for the model
            context, source_metadata = self._build_context(statute_matches, caselaw_matches)

            if not context.strip():
                return "I'm sorry, I could not find relevant legal information for your query. Please try rephrasing your question or consult a legal professional."

            # 4. Generate a response based on the context
            answer = self._generate_legal_response(
                question, context,
                has_statutes=bool(statute_matches),
                has_caselaw=bool(caselaw_matches)
            )

            # 5. Format the final output with sources and a disclaimer
            sources_section = self._format_sources(source_metadata)

            return answer + sources_section

        except Exception as e:
            logger.error(f"An unexpected error occurred in get_wakili_response: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred while processing your request. Please try again."


def main():
    """Main function to run the interactive legal assistant in the console."""
    try:
        wakili = WakiliAI()
        print("\n--- Wakili AI Enhanced Legal Assistant is ready! ---")
        print("Ask your legal questions in plain English. Type 'exit' to quit.")

        while True:
            user_question = input("\nAsk Wakili AI a question: ").strip()

            if user_question.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye! Thank you for using Wakili AI.")
                break

            if not user_question:
                print("Please enter a valid question.")
                continue

            print("\n-> Thinking... ⏳")

            try:
                final_response = wakili.get_wakili_response(user_question)
                print(f"\n{final_response}")
            except Exception as e:
                logger.error(f"Error while processing question: '{user_question}': {e}", exc_info=True)
                print("\nI'm sorry, I encountered an error. Please try asking your question again.")

    except Exception as e:
        logger.error(f"Failed to initialize Wakili AI: {e}", exc_info=True)
        print("\n--- FATAL: Could not start Wakili AI due to an initialization error. Please check the logs. ---")


if __name__ == "__main__":
    main()