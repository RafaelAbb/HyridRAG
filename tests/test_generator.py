import json
from unittest.mock import MagicMock, patch

import pytest

from src.generation.base import CitationVerification, GenerationResult, JudgeEnum
from src.generation.generator import (
    calculated_confidence,
    judge_one_citation,
    parse_response,
    generate_answer,
)
from src.retrieval.base import RetrievalResult


# ── helpers ───────────────────────────────────────────────────────────────────

def make_retrieval_result(doc_id: str, score: float = 0.9, text: str = "chunk text") -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, metadata=None, text=text, score=score)


def make_generation_result(
    answer: str = "The answer is X [1].",
    claim_source_pairs=None,
    has_answer: bool = True,
    confidence: float = 0.0,
    references=None,
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        claim_source_pairs=claim_source_pairs or [("The answer is X", "doc_1")],
        has_answer=has_answer,
        confidence=confidence,
        list_of_references=references or [make_retrieval_result("doc_1")],
    )


def mock_openai_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ── parse_response ────────────────────────────────────────────────────────────

def test_parse_response_valid_json():
    raw = json.dumps({
        "answer": "The sky is blue [1].",
        "claim_source_pairs": [["The sky is blue", "doc_1"]],
        "has_answer": True,
    })
    result = parse_response(raw)
    assert result.answer == "The sky is blue [1]."
    assert result.has_answer is True
    assert result.claim_source_pairs == [["The sky is blue", "doc_1"]]


def test_parse_response_no_answer():
    raw = json.dumps({
        "answer": "I don't know.",
        "claim_source_pairs": [],
        "has_answer": False,
    })
    result = parse_response(raw)
    assert result.has_answer is False
    assert result.claim_source_pairs == []


def test_parse_response_invalid_json_returns_empty_result():
    result = parse_response("this is not json")
    assert result.has_answer is False
    assert result.answer == ""
    assert result.confidence == 0.0


# ── calculated_confidence ─────────────────────────────────────────────────────

def test_confidence_zero_when_no_answer():
    result = make_generation_result(has_answer=False)
    assert calculated_confidence(result) == 0.0


def test_confidence_lower_when_no_references():
    result_with = make_generation_result(references=[make_retrieval_result("doc_1", score=1.0)])
    result_without = make_generation_result(references=[])
    assert calculated_confidence(result_without) < calculated_confidence(result_with)


def test_confidence_full_when_all_citations_supported_and_high_score():
    refs = [make_retrieval_result("doc_1", score=1.0)]
    result = make_generation_result(
        claim_source_pairs=[("claim A", "doc_1")],
        references=refs,
    )
    confidence = calculated_confidence(result)
    # retrieval = 1.0, citation_coverage = 1.0 → 0.5*1 + 0.5*1 = 1.0
    assert confidence == pytest.approx(1.0)


def test_confidence_partial_when_some_sources_missing():
    refs = [make_retrieval_result("doc_1", score=1.0)]
    result = make_generation_result(
        claim_source_pairs=[("claim A", "doc_1"), ("claim B", "")],
        references=refs,
    )
    confidence = calculated_confidence(result)
    # citation_coverage = 0.5, retrieval = 1.0 → 0.75
    assert confidence == pytest.approx(0.75)


def test_confidence_between_zero_and_one():
    refs = [make_retrieval_result("doc_1", score=0.6)]
    result = make_generation_result(references=refs)
    confidence = calculated_confidence(result)
    assert 0.0 <= confidence <= 1.0


# ── judge_one_citation ────────────────────────────────────────────────────────

def test_judge_one_citation_returns_supported():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response("supported")
    result = judge_one_citation(("The sky is blue", "doc_1"), "The sky is blue.", mock_client)
    assert result == JudgeEnum.SUPPORTED


def test_judge_one_citation_returns_not_supported():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response("not_supported")
    result = judge_one_citation(("The sky is green", "doc_1"), "The sky is blue.", mock_client)
    assert result == JudgeEnum.NOT_SUPPORTED


def test_judge_one_citation_invalid_response_defaults_to_insufficient_info():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response("maybe")
    result = judge_one_citation(("some claim", "doc_1"), "some chunk", mock_client)
    assert result == JudgeEnum.INSUFFICIENT_INFO


def test_judge_one_citation_case_insensitive():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response("  SUPPORTED  ")
    result = judge_one_citation(("claim", "doc_1"), "chunk", mock_client)
    assert result == JudgeEnum.SUPPORTED


# ── generate_answer ───────────────────────────────────────────────────────────

def test_generate_answer_returns_generation_result():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response(json.dumps({
        "answer": "Photosynthesis converts light to energy [1].",
        "claim_source_pairs": [["Photosynthesis converts light to energy", "doc_1"]],
        "has_answer": True,
    }))
    chunks = [make_retrieval_result("doc_1", score=0.85, text="Plants convert light to energy.")]
    result = generate_answer("What is photosynthesis?", chunks, openai_client=mock_client)
    assert isinstance(result, GenerationResult)
    assert result.has_answer is True
    assert result.answer != ""
    assert result.list_of_references == chunks


def test_generate_answer_no_answer_has_zero_confidence():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response(json.dumps({
        "answer": "I don't know.",
        "claim_source_pairs": [],
        "has_answer": False,
    }))
    chunks = [make_retrieval_result("doc_1")]
    result = generate_answer("What is the meaning of life?", chunks, openai_client=mock_client)
    assert result.has_answer is False
    assert result.confidence == 0.0
