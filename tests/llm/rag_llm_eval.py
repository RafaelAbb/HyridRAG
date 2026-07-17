from deepeval.dataset import Golden, EvaluationDataset

from retrieval.fusion import hybrid_retrieve
from src.generation.generator import generate_answer
from tests.test_deepeval_eval import reranker

goldens = [
    Golden(input="What is your name?"),
    Golden(input="Choose a number between 1 and 100"),
    # ...
]

dataset = EvaluationDataset(goldens=goldens)


import asyncio
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import TaskCompletionMetric
...

@observe()
async def my_ai_agent(query: str) -> str:
    answer = await generate_answer(query)
    update_current_trace(input=query, output=answer)
    return answer

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric()]):
    task = asyncio.create_task(my_ai_agent(golden.input))
    dataset.evaluate(task)