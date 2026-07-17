import logging
from src.config import settings
from openai import OpenAI

from src.retrieval.base import RetrievalResult

from src.generation.base import CitationVerification, GenerationResult, JudgeEnum
import json


SYSTEM_PROMPT = """
role: You are a highly intelligent and knowledgeable assistant.
goal: Your task is to provide accurate and concise answers to the user based on the information provided in the retrieved results.
context: 
- You will be given a list of retrieved results, each containing a claim and its source.
- Return I don't know if context is insufficient.
restrictions:
- You must only use the information provided in the retrieved results to answer the user's question.
output_format:
- Your answer should be a single, structured response formated in json as the following:
{
    "answer": "Your concise answer here, where for each sentence use the id of the source that supports it in square brackets, e.g., [source_id].",
    "claim_source_pairs": [ list of Tuple{claim, source_index} ],
    "has_answer": true if an answer is provided, false otherwise,
}
"""

JUDGE_PROMPT = """
You are a citation verifier. You will be given a CLAIM and a CHUNK of text.
Your task: determine if the chunk directly supports the claim.

Rules:
- Answer ONLY with one of: supported, not_supported, insufficient_info
- Do not explain. Do not add any other text.
- "supported" means the chunk contains clear evidence for the claim.
- "not_supported" means the chunk contradicts or is irrelevant to the claim.
- "insufficient_info" means the chunk is related but does not clearly confirm or deny the claim.
"""

def calculated_confidence(generation_result: GenerationResult,) -> float:
    """
    Calculate the confidence score based on the presence of an answer and the number of claim-source pairs.

    Args:
        generation_result (GenerationResult): The generation result object.

    Returns:
        float: A confidence score between 0.0 and 1.0.
    """
    if not generation_result.has_answer:
        return 0.0
    
    # retrieval signal: average score of the chunks passed in
    retrieval_score = sum(r.score for r in generation_result.list_of_references) / len(generation_result.list_of_references) if generation_result.list_of_references else 0.0

    # citation coverage: fraction of claims that have a source (non-empty second element)
    pairs = generation_result.claim_source_pairs
    if pairs:
        citation_coverage = sum(1 for _, src in pairs if src) / len(pairs)
    else:
        citation_coverage = 0.0

    return 0.5 * retrieval_score + 0.5 * citation_coverage
    





def parse_response(response: str) -> GenerationResult:
    """
    Parse the response from the model and extract the answer and claim-source pairs.

    Args:
        response (str): The response from the model.

    Returns:
        GenerationResult: An object containing the parsed answer and claim-source pairs.
    """
    try:
        # Assuming the response is in JSON format
        parsed = json.loads(response)
        
        answer = parsed.get("answer", "")
        claim_source_pairs = parsed.get("claim_source_pairs", [])
        has_answer = parsed.get("has_answer", True)

        return GenerationResult(answer=answer, claim_source_pairs=claim_source_pairs, has_answer=has_answer)
    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON
        return GenerationResult(answer="", claim_source_pairs=[], has_answer=False, confidence=0.0)#todo: send to another llm to only fix the json format and return the fixed json to this function


def generate_answer(query: str, 
                retrieved_results: list[RetrievalResult], 
                openai_client: OpenAI | None = None) -> GenerationResult:
    
    openai = openai_client or OpenAI(api_key=settings.openai_api_key)
    
    context_block = "\n".join(f"[{i}] {result.text}" for i, result in enumerate(retrieved_results, start = 1))
    
    completion = openai.chat.completions.create(
                model=settings.generation_model,
                temperature=settings.generation_temperature,
                max_tokens=settings.generation_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":  f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"}
                ]
    )
    
    parsed_result = parse_response(completion.choices[0].message.content)
    
    parsed_result.list_of_references = retrieved_results
    parsed_result.confidence = calculated_confidence(parsed_result)
    
    return parsed_result


def judge_one_citation(claim_source: tuple[str, str], chunk_text: str, openai_client: OpenAI = None) -> JudgeEnum:
    # one cheap LLM call, returns True/False
    openai_client = openai_client or OpenAI(api_key=settings.openai_api_key)
    claim, source = claim_source
    
    
    completion = openai_client.chat.completions.create(
                model=settings.judgement_model,
                temperature=settings.judgement_temperature,
                max_tokens=settings.judgement_max_tokens,
                messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content":  f"CONTEXT:\n{chunk_text}\n\nCLAIM: {claim}"}
                ]
    )
    judgment = completion.choices[0].message.content.strip().lower()
    try:
        judge_result = JudgeEnum(judgment)
    except Exception as e:
        logging.error(f"Error parsing judge result: {e}. Response was: {judgment}, claim: {claim}, source: {source}")
        judge_result = JudgeEnum.INSUFFICIENT_INFO  # Default to insufficient info on error
    return judge_result 


def judge_citations(claim_source_pairs: list[tuple[str, str]], retrieved_results: list[RetrievalResult], openai_client: OpenAI = None) -> list[CitationVerification]:

    openai_client = openai_client or OpenAI(api_key=settings.openai_api_key)
    # generate_answer's context_block is numbered "[1] ... [2] ..." (1-based position,
    # not doc_id) and SYSTEM_PROMPT tells the model to cite that number — so citations
    # must be resolved by position in retrieved_results, not by doc_id.
    index_dict = {str(i): result for i, result in enumerate(retrieved_results, start=1)}
    citation_verifications = []

    for claim, source in claim_source_pairs:
        source = str(source)
        if source not in index_dict:
            logging.error(f"Source {source} not found in retrieved results.")
        else:
            result = index_dict[source]
            judge_result = judge_one_citation((claim, source), result.text, openai_client)
            citation_verifications.append(
                CitationVerification(
                    claim=claim,
                    doc_id=result.doc_id,
                    is_supported=judge_result == JudgeEnum.SUPPORTED,
                )
            )
    return citation_verifications
