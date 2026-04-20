import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeEmbeddingsAPI:
    def create(self, *, model: str, input: list[str]):
        data = []
        for index, text in enumerate(input):
            length = float(len(text))
            checksum = float(sum(ord(char) for char in text) % 97)
            data.append(
                SimpleNamespace(
                    index=index,
                    embedding=[length, checksum, float(index + 1)],
                )
            )
        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


@pytest.fixture
def fake_openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()
