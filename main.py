import os
import sys
import traceback
from io import StringIO

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


# Load environment variables
load_dotenv()


# -----------------------------
# AI Pipe Client
# -----------------------------

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI()


# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request Model
# -----------------------------

class CodeRequest(BaseModel):
    code: str


# -----------------------------
# AI Response Model
# -----------------------------

class ErrorAnalysis(BaseModel):
    error_lines: list[int]


# -----------------------------
# Execute Python Code
# -----------------------------

def execute_python_code(code: str) -> dict:

    old_stdout = sys.stdout

    sys.stdout = StringIO()

    try:

        exec(code)

        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:

        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:

        sys.stdout = old_stdout


# -----------------------------
# AI Error Analysis
# -----------------------------

def analyze_error_with_ai(code: str, traceback_output: str) -> list[int]:

    prompt = f"""
Analyze this Python code and its error traceback.

Your task is to identify the exact line number(s) in the
ORIGINAL Python code where the error occurred.

Rules:
1. Use the original source code line numbers.
2. Do not invent line numbers.
3. Ignore internal Python/library traceback lines.
4. Return only the line numbers responsible for the error.
5. Return the result as JSON using the field error_lines.

CODE:
{code}

TRACEBACK:
{traceback_output}
"""

    response = client.chat.completions.create(

        model="openai/gpt-4.1-nano",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


# -----------------------------
# Root Endpoint
# -----------------------------

@app.get("/")
def root():
    return {
        "status": "ok"
    }


# -----------------------------
# Code Interpreter Endpoint
# -----------------------------

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # If code executed successfully,
    # DO NOT call AI.
    if execution["success"]:

        return {
            "error": [],
            "result": execution["output"]
        }

    # If code failed,
    # ask AI to identify error lines.
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }