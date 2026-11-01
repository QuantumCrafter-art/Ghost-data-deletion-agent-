


import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

root = Path(__file__).resolve().parent
load_dotenv(root / ".env", override=False)
load_dotenv(root / "code" / ".env", override=False)

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY not found")

client = genai.Client(api_key=key)

tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="lookup_evidence",
        description="Look up deletion evidence for a case.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"case_id": types.Schema(type=types.Type.STRING)},
            required=["case_id"],
        ),
    )
])

candidates = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
]

for model in candidates:
    print(f"\n--- {model} ---")

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly MODEL_WORKS",
        )
        print("Generation: PASS |", (response.text or "")[:100])
    except Exception as e:
        print(f"Generation: FAIL | {type(e).__name__}: {e}")
        continue

    try:
        response = client.models.generate_content(
            model=model,
            contents="Call lookup_evidence with case_id CASE-001.",
            config=types.GenerateContentConfig(tools=[tool]),
        )
        calls = [
            part.function_call.name
            for candidate in (response.candidates or [])
            for part in (candidate.content.parts or [])
            if getattr(part, "function_call", None)
        ]
        print("Function calling:", "PASS" if "lookup_evidence" in calls else "NOT CONFIRMED")
    except Exception as e:
        print(f"Function calling: FAIL | {type(e).__name__}: {e}")
