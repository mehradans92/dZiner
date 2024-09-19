from langchain.schema import AIMessage, ChatResult, ChatGeneration
from langchain.chat_models.base import BaseChatModel

from typing import List
from pydantic import PrivateAttr

# This is just a dummy fake chat model for testing
class FakeChatModel(BaseChatModel):
    _responses: List[str] = PrivateAttr()
    _call_count: int = PrivateAttr(0)

    def __init__(self, responses):
        super().__init__()
        object.__setattr__(self, '_responses', responses)
        object.__setattr__(self, '_call_count', 0)

    @property
    def _llm_type(self):
        return "fake-chat-model"

    def _generate(self, messages, stop=None, **kwargs):
        response = self._responses[self._call_count % len(self._responses)]
        object.__setattr__(self, '_call_count', self._call_count + 1)
        ai_message = AIMessage(content=response)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    async def _agenerate(self, messages, stop=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)
