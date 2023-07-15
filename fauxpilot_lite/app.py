"""REST endpoints as used by GitHub copilot"""
import logging
import json
import os
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr
from sse_starlette.sse import EventSourceResponse  # required for stream response

from fauxpilot_lite.log_config import uvicorn_logger  # verbose logging for uvicorn HTTP
from fauxpilot_lite.model import generate

# set up logging
logging.config.dictConfig(uvicorn_logger)

# Defining the POST body format
# It's named "OpenAIinput" because GitHub copilot was developed by OpenAI
# Those attributes without a default are required from the POST body, notably "model" is the FauxPilot addition
class OpenAIinput(BaseModel):
    model: str = "codegen2"
    prompt: Optional[str]
    suffix: Optional[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, list]] = ['\n']
    presence_penalty: Optional[float] = 0.
    frequency_penalty: Optional[float] = 1.
    best_of: Optional[int] = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

# Exception object to raise on HTTP/400
class FauxPilotException(Exception):
    def __init__(self, message: str, error_type: Optional[str] = None, param: Optional[str] = None,
                 code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

    def json(self):
        return {
            'error': {
                'message': self.message,
                'type': self.error_type,
                'param': self.param,
                'code': self.code
            }
        }


# FastAPI and endpoints below
app = FastAPI(
    title="FauxPilot-Lite",
    description="The minimal implementation of a self-hosted version of GitHub copilot.",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


@app.exception_handler(FauxPilotException)
async def fauxpilot_handler(request: Request, exc: FauxPilotException):
    """Return HTTP/400 on error"""
    logger = logging.getLogger("pilot.exc")
    logger.error(exc)
    return JSONResponse(
        status_code=400,
        content=exc.json()
    )


@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    """Return a dummy token that never expires
    Required to support copilot.vim
    """
    content = {'token': '1', 'expires_at': 2600000000, 'refresh_in': 900}
    return JSONResponse(
        status_code=200,
        content=content
    )


@app.post("/v1/engines/{modelname}/completions")
@app.post("/v1/engines/copilot-codex/completions") # to support copilot.vim
@app.post("/v1/completions")
async def completions(data: OpenAIinput, modelname: str = "codegen2"):
    """Main function for handing copilot I/O

    Example POST body, as provided by the argument `data`:
        {
         'model': 'fastertransformer',
         'prompt': 'import os\nimport math\n\ndef black_sch',
         'suffix': None,
         'max_tokens': 200,
         'temperature': 0.1,
         'top_p': 1.0,
         'n': 1,
         'stream': None,
         'logprobs': None,
         'echo': None,
         'stop': ['\n'],
         'presence_penalty': 0,
         'frequency_penalty': 1,
         'best_of': 1,
         'logit_bias': None,
         'user': None
        }

    Example response body:
        {
         "id": "cmpl-s2F85JSHk7fL6s62nm2UsoD0yfjYL",
         "model": "codegen",
         "object": "text_completion",
         "created": 1689085524,
         "choices": [
           {
            "text": "oles(s, r, rt, t, X):",
            "index": 0,
            "finish_reason": "stop",
            "logprobs": null
           }
         ],
         "usage": {
            "completion_tokens": 14,
            "prompt_tokens": 11,
            "total_tokens": 25
         }
        }
    """
    logger = logging.getLogger("pilot")
    logger.setLevel(logging.DEBUG)  # adjust for verbosity
    data = data.dict()
    logger.info(json.dumps(data, indent=4, ensure_ascii=False))
    try:
        data['model'] = modelname
        content = generate(data=data)
    except Exception as ex:
        raise FauxPilotException(
            message=str(ex),
            error_type="invalid_request_error",
            param=None,
            code=None,
        )
    jsoncontent = json.dumps(content, indent=4, ensure_ascii=False)
    logger.info(jsoncontent)

    if data.get("stream") is not None:
        return EventSourceResponse(
            content=iter([jsoncontent, '[DONE]']),
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=jsoncontent,
            media_type="application/json"
        )


if __name__ == "__main__":
    uvicorn.run("fauxpilot_lite.app:app", host="0.0.0.0", port=5000)
