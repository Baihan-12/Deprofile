from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.deprofile_service import DEFAULT_CONFIG, chat_with_profile, get_demo_status, get_profile_detail, list_profile_summaries, validate_config
from api.models import ChatRequest, ValidateConfigRequest

app = FastAPI(title='Deprofile UI API', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/config/defaults')
def config_defaults() -> dict[str, dict[str, str]]:
    return {'config': {**DEFAULT_CONFIG, 'apiKey': ''}}


@app.post('/api/config/validate')
def config_validate(payload: ValidateConfigRequest) -> dict[str, object]:
    return validate_config(payload.config.model_dump(), session_id=payload.sessionId or '')


@app.get('/api/demo/status')
def demo_status(session_id: str = '') -> dict[str, object]:
    return get_demo_status(session_id=session_id)


@app.get('/api/profiles')
def profiles(source: str = 'selected_samples') -> dict[str, object]:
    if source not in {'selected_samples', 'complete_index'}:
        raise HTTPException(status_code=400, detail='Unsupported source')
    return {'items': list_profile_summaries(source)}


@app.get('/api/profiles/{pair_id}')
def profile_detail(pair_id: str, source: str = 'selected_samples', language: str = 'en') -> dict[str, object]:
    try:
        return get_profile_detail(source, pair_id, language=language)
    except KeyError as error:
        raise HTTPException(status_code=404, detail='Profile not found') from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail='Timeline file missing') from error


@app.post('/api/chat')
def chat(payload: ChatRequest) -> dict[str, object]:
    config_check = validate_config(payload.config.model_dump(), session_id=payload.sessionId or '')
    if not config_check['ok']:
        raise HTTPException(status_code=400, detail=config_check['message'])
    try:
        return chat_with_profile(
            config=payload.config.model_dump(),
            source=payload.source,
            pair_id=payload.pairId,
            messages=[message.model_dump() for message in payload.messages],
            language=payload.language,
            session_id=payload.sessionId or '',
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail='Profile not found') from error
    except PermissionError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
