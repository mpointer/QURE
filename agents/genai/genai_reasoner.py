"""
GenAI Reasoner Agent

LLM-powered reasoning with RAG, chain-of-thought, and citation linking.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import openai
from anthropic import Anthropic

from common.schemas import GenAIRequest, GenAIResponse, TextSpan

# Aliases for backward compatibility
GenAIReasoningRequest = GenAIRequest
GenAIReasoningResponse = GenAIResponse

try:
    from substrate import get_evidence_tracker, get_vector_store
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False

logger = logging.getLogger(__name__)


class GenAIReasoner:
    """
    GenAI Reasoner Agent for LLM-powered analysis

    Responsibilities:
    - LLM reasoning (OpenAI GPT-4, Anthropic Claude)
    - RAG with semantic search
    - Chain-of-thought prompting
    - Citation extraction and linking
    - Structured output parsing
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo-preview",
    ):
        """
        Initialize GenAI Reasoner

        Args:
            provider: "openai" or "anthropic"
            model: Model name
        """
        self.provider = provider
        self.model = model

        # Initialize LLM clients
        if provider == "openai":
            self.openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            self.anthropic_client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Knowledge substrate connections (optional)
        if SUBSTRATE_AVAILABLE:
            self.vector_store = get_vector_store()
            self.evidence_tracker = get_evidence_tracker()
        else:
            self.vector_store = None
            self.evidence_tracker = None

        logger.info(
            f"✅ GenAI Reasoner initialized with {provider}/{model}"
        )

    def reason(
        self,
        request: GenAIReasoningRequest,
    ) -> GenAIReasoningResponse:
        """
        Perform LLM reasoning with RAG

        Args:
            request: GenAIReasoningRequest with query and context

        Returns:
            GenAIReasoningResponse with answer and citations
        """
        query = request.query
        case_id = request.case_id
        use_rag = request.use_rag
        temperature = request.temperature

        try:
            # Retrieve relevant documents if RAG enabled
            context_documents = []
            if use_rag:
                context_documents = self._retrieve_context(query, top_k=5)

            # Build prompt
            prompt = self._build_prompt(
                query=query,
                context_documents=context_documents,
                system_instructions=request.system_instructions,
            )

            # Call LLM
            answer, usage_tokens = self._call_llm(
                prompt=prompt,
                temperature=temperature,
            )

            # Extract citations
            citations = self._extract_citations(
                answer=answer,
                context_documents=context_documents,
            )

            # Store evidence
            if citations:
                for citation in citations:
                    self.evidence_tracker.add_citation(
                        case_id=case_id,
                        claim_text=citation["claim_text"],
                        source_id=citation["source_id"],
                        source_span=citation.get("source_span"),
                    )

            # Compute confidence (simple heuristic based on citation count)
            confidence = min(1.0, len(citations) * 0.2) if citations else 0.5

            explanation = f"Generated answer with {len(citations)} citations"

            logger.debug(
                f"GenAI reasoning for case {case_id}: {len(answer)} chars, "
                f"{len(citations)} citations, {usage_tokens} tokens"
            )

            return GenAIReasoningResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                answer=answer,
                citations=citations,
                confidence=confidence,
                usage_tokens=usage_tokens,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"GenAI reasoning failed for case {case_id}: {e}")
            return GenAIReasoningResponse(
                case_id=case_id,
                from_agent=request.to_agent or request.from_agent,
                to_agent=None,
                answer="",
                citations=[],
                confidence=0.0,
                usage_tokens=0,
                explanation=f"Reasoning error: {str(e)}",
            )

    def _retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using semantic search

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of document dicts
        """
        try:
            # Get query embedding
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode(query).tolist()

            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=top_k,
            )

            documents = []
            for i in range(len(results["ids"][0])):
                documents.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

            logger.debug(f"Retrieved {len(documents)} context documents")
            return documents

        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return []

    def _build_prompt(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_instructions: Optional[str] = None,
    ) -> str:
        """
        Build LLM prompt with RAG context

        Args:
            query: User query
            context_documents: Retrieved documents
            system_instructions: Optional system prompt

        Returns:
            Full prompt string
        """
        prompt_parts = []

        # System instructions
        if system_instructions:
            prompt_parts.append(f"SYSTEM: {system_instructions}\n")
        else:
            prompt_parts.append(
                "SYSTEM: You are an expert analyst helping resolve back-office exceptions. "
                "Provide clear, accurate answers with citations to support your reasoning.\n"
            )

        # Context documents
        if context_documents:
            prompt_parts.append("\nCONTEXT DOCUMENTS:\n")
            for i, doc in enumerate(context_documents, 1):
                prompt_parts.append(
                    f"[{i}] (ID: {doc['id']})\n{doc['content']}\n"
                )

        # Query
        prompt_parts.append(f"\nQUERY: {query}\n")

        # Instructions for citations
        prompt_parts.append(
            "\nProvide your answer with inline citations in the format [doc_id]. "
            "For example: 'The transaction amount was $1,234.56 [doc_123].'\n"
        )

        return "\n".join(prompt_parts)

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> tuple[str, int]:
        """
        Call LLM API

        Args:
            prompt: Full prompt
            temperature: Sampling temperature

        Returns:
            Tuple of (answer, usage_tokens)
        """
        if self.provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000,
            )

            answer = response.choices[0].message.content
            usage_tokens = response.usage.total_tokens

            return answer, usage_tokens

        elif self.provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            answer = response.content[0].text
            usage_tokens = response.usage.input_tokens + response.usage.output_tokens

            return answer, usage_tokens

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _extract_citations(
        self,
        answer: str,
        context_documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from LLM answer

        Args:
            answer: LLM-generated answer
            context_documents: Context documents

        Returns:
            List of citation dicts
        """
        import re

        citations = []

        # Find all inline citations [doc_id]
        citation_pattern = r"\[([^\]]+)\]"
        matches = re.finditer(citation_pattern, answer)

        # Build doc_id to document mapping
        doc_map = {doc["id"]: doc for doc in context_documents}

        for match in matches:
            doc_id = match.group(1)

            if doc_id in doc_map:
                # Extract claim text (sentence containing citation)
                claim_start = max(0, match.start() - 100)
                claim_end = min(len(answer), match.end() + 100)
                claim_text = answer[claim_start:claim_end].strip()

                # Get source document
                source_doc = doc_map[doc_id]

                # Create text span (simplified - full implementation would find exact match)
                source_span = TextSpan(
                    start_char=0,
                    end_char=len(source_doc["content"]),
                    text=source_doc["content"][:200],  # First 200 chars
                )

                citations.append({
                    "claim_text": claim_text,
                    "source_id": doc_id,
                    "source_span": source_span,
                })

        logger.debug(f"Extracted {len(citations)} citations from answer")
        return citations

    def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 500,
    ) -> str:
        """
        Generate summary of multiple documents

        Args:
            documents: List of document dicts
            max_length: Max summary length in words

        Returns:
            Summary text
        """
        try:
            # Build prompt
            doc_texts = [f"Document {i+1}:\n{doc['content']}\n"
                        for i, doc in enumerate(documents)]

            prompt = (
                f"Summarize the following {len(documents)} documents in {max_length} words or less:\n\n"
                + "\n".join(doc_texts)
            )

            # Call LLM
            summary, _ = self._call_llm(prompt, temperature=0.3)

            return summary

        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return ""

    def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using LLM

        Args:
            text: Input text
            schema: JSON schema for desired output

        Returns:
            Extracted data dict
        """
        try:
            import json

            # Build prompt
            prompt = (
                f"Extract the following information from the text below:\n\n"
                f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
                f"TEXT:\n{text}\n\n"
                f"Return a JSON object matching the schema."
            )

            # Call LLM
            response, _ = self._call_llm(prompt, temperature=0.0)

            # Parse JSON
            # Find JSON block in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group(0))
                return extracted_data
            else:
                logger.warning("No JSON found in LLM response")
                return {}

        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {}

    def chain_of_thought(
        self,
        problem: str,
        steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Chain-of-thought reasoning

        Args:
            problem: Problem statement
            steps: Optional reasoning steps

        Returns:
            Dict with reasoning chain and final answer
        """
        try:
            # Build prompt
            prompt = f"Problem: {problem}\n\n"

            if steps:
                prompt += "Follow these reasoning steps:\n"
                for i, step in enumerate(steps, 1):
                    prompt += f"{i}. {step}\n"
                prompt += "\n"

            prompt += (
                "Think through this step-by-step. Show your reasoning for each step, "
                "then provide a final answer.\n\n"
                "Format your response as:\n"
                "Step 1: [reasoning]\n"
                "Step 2: [reasoning]\n"
                "...\n"
                "Final Answer: [answer]"
            )

            # Call LLM
            response, tokens = self._call_llm(prompt, temperature=0.2)

            # Parse response
            import re
            step_pattern = r"Step \d+: (.+?)(?=Step \d+:|Final Answer:|$)"
            answer_pattern = r"Final Answer: (.+)"

            reasoning_steps = re.findall(step_pattern, response, re.DOTALL)
            answer_match = re.search(answer_pattern, response, re.DOTALL)

            final_answer = answer_match.group(1).strip() if answer_match else ""

            return {
                "problem": problem,
                "reasoning_steps": [s.strip() for s in reasoning_steps],
                "final_answer": final_answer,
                "usage_tokens": tokens,
            }

        except Exception as e:
            logger.error(f"Chain-of-thought reasoning failed: {e}")
            return {
                "problem": problem,
                "reasoning_steps": [],
                "final_answer": "",
                "error": str(e),
            }


# Singleton instance
_genai_reasoner: Optional[GenAIReasoner] = None


def get_genai_reasoner(
    provider: str = "openai",
    model: str = "gpt-4-turbo-preview",
) -> GenAIReasoner:
    """
    Get or create singleton GenAIReasoner instance

    Args:
        provider: LLM provider
        model: Model name

    Returns:
        GenAIReasoner instance
    """
    global _genai_reasoner

    if _genai_reasoner is None:
        _genai_reasoner = GenAIReasoner(provider=provider, model=model)

    return _genai_reasoner
