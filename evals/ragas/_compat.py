"""Import this before anything from `ragas` in this package.

ragas==0.4.3 hard-imports langchain_community.chat_models.vertexai and
langchain_community.llms.VertexAI at module load time (ragas/llms/base.py),
just to reference the classes in an internal type list. Both were removed
from langchain-community>=0.4 as part of its integration-package sunset.
We only ever use the OpenAI provider, so we never need Vertex AI — this
stubs the two missing symbols in sys.modules before ragas imports them,
instead of downgrading langchain-community (which would cascade into
downgrading langchain-text-splitters, a real dependency of src/ingestion/).
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake_chat_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
    _fake_chat_vertexai.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_chat_vertexai

import langchain_community.llms as _llms  # noqa: E402

if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})
