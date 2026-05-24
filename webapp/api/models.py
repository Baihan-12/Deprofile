from typing import Literal

from pydantic import BaseModel, Field


class ApiConfig(BaseModel):
    apiKey: str = Field(default='')
    model: str = Field(default='gemini-3-pro-preview')
    baseUrl: str = Field(default='https://aidp.bytedance.net/api/modelhub/online/v2/crawl')
    apiVersion: str | None = Field(default='2024-02-01')
    apiType: str | None = Field(default='azure')


class ChatTurn(BaseModel):
    role: Literal['user', 'assistant']
    content: str


class ChatRequest(BaseModel):
    config: ApiConfig
    source: Literal['selected_samples', 'complete_index'] = 'selected_samples'
    pairId: str
    messages: list[ChatTurn]
    language: Literal['en'] = 'en'
    sessionId: str | None = Field(default='')


class ValidateConfigRequest(BaseModel):
    config: ApiConfig
    sessionId: str | None = Field(default='')
