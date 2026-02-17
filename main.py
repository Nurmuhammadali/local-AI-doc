"""
LocalAIdoc - Local AI Documentation Generator
Inspired by superdocs.cloud - runs 100% free & locally
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import requests
import json
import os
import pathlib
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LocalAIdoc", description="AI-powered documentation generator")
templates = Jinja2Templates(directory="templates")

# ─── File filters ────────────────────────────────────────────────────────────
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
    '.cpp', '.c', '.cs', '.rb', '.php', '.swift', '.kt', '.vue',
    '.svelte', '.dart', '.scala', '.r', '.lua', '.sh', '.bash',
    '.html', '.css', '.scss', '.sass'
}
CONFIG_EXTENSIONS = {
    '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.cfg',
    '.ini', '.env.example', '.xml', '.dockerfile'
}
SKIP_DIRS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.next', 'vendor', '.idea', '.vscode',
    'coverage', '.pytest_cache', 'target', 'bin', 'obj'
}
MAX_FILE_SIZE = 8_000    # characters per file
MAX_FILES = 30           # max files to send to AI
MAX_TOTAL_CHARS = 28_000 # total characters cap for prompt (~7k tokens)


# ─── GitHub file reader ───────────────────────────────────────────────────────
def get_github_files(repo_url: str) -> dict:
    """Fetch files from a public GitHub repository using GitHub API."""
    url = repo_url.strip().rstrip('/')
    # Strip github.com prefix
    for prefix in ['https://github.com/', 'http://github.com/', 'github.com/']:
        url = url.replace(prefix, '')
    parts = url.split('/')
    if len(parts) < 2:
        return {'__error__': 'Invalid GitHub URL. Expected: https://github.com/owner/repo'}

    owner, repo = parts[0], parts[1].replace('.git', '')
    headers = {'Accept': 'application/vnd.github.v3+json'}

    # Try main then master branch
    tree_data = None
    for branch in ['HEAD', 'main', 'master']:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            tree_data = resp.json()
            break

    if not tree_data:
        return {'__error__': f'Cannot access repo "{owner}/{repo}". Make sure it is public.'}

    files = {}
    total_chars = 0

    for item in tree_data.get('tree', []):
        if item['type'] != 'blob':
            continue
        path = item['path']
        ext = pathlib.Path(path).suffix.lower()

        # Skip unwanted files/dirs
        path_parts = pathlib.Path(path).parts
        if any(skip in path_parts for skip in SKIP_DIRS):
            continue
        if ext not in CODE_EXTENSIONS and ext not in CONFIG_EXTENSIONS:
            continue
        if item.get('size', 0) > 80_000:
            continue

        # Fetch raw content
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        try:
            content_resp = requests.get(raw_url, timeout=10)
            if content_resp.status_code == 200:
                content = content_resp.text[:MAX_FILE_SIZE]
                files[path] = content
                total_chars += len(content)
        except Exception:
            continue

        if len(files) >= MAX_FILES or total_chars >= MAX_TOTAL_CHARS:
            break

    return files


# ─── Local folder reader ──────────────────────────────────────────────────────
def get_local_files(folder_path: str) -> dict:
    """Read source files from a local folder."""
    base = pathlib.Path(folder_path)
    if not base.exists():
        return {'__error__': f'Folder not found: {folder_path}'}
    if not base.is_dir():
        return {'__error__': f'Path is not a directory: {folder_path}'}

    files = {}
    total_chars = 0

    for file_path in sorted(base.rglob('*')):
        if not file_path.is_file():
            continue
        if any(skip in file_path.parts for skip in SKIP_DIRS):
            continue
        ext = file_path.suffix.lower()
        if ext not in CODE_EXTENSIONS and ext not in CONFIG_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:MAX_FILE_SIZE]
            relative = str(file_path.relative_to(base))
            files[relative] = content
            total_chars += len(content)
        except Exception:
            continue

        if len(files) >= MAX_FILES or total_chars >= MAX_TOTAL_CHARS:
            break

    return files


# ─── Prompt builder ───────────────────────────────────────────────────────────
def build_prompt(files: dict, doc_type: str, project_name: str = "") -> str:
    """Build the AI documentation prompt."""
    # Build file summary
    file_list = "\n".join(f"  - {p}" for p in list(files.keys())[:50])
    file_content = ""
    total = 0
    for path, content in files.items():
        chunk = f"\n\n### {path}\n```\n{content}\n```"
        if total + len(chunk) > MAX_TOTAL_CHARS:
            break
        file_content += chunk
        total += len(chunk)

    proj = project_name or "this project"

    prompts = {
        'readme': f"""You are a senior technical writer. Analyze the following codebase for "{proj}" and write a comprehensive, professional README.md.

FILE STRUCTURE:
{file_list}

CODEBASE:
{file_content}

Write a complete README.md with these sections:
# {proj}

## Description
(Clear 2-3 sentence description of what the project does)

## Features
(Bullet list of key features found in the code)

## Tech Stack
(Technologies, frameworks, languages used)

## Installation
(Step-by-step installation commands)

## Usage
(How to run and use the project with examples)

## Project Structure
(Directory tree and explanation)

## API / Configuration
(If applicable, key configuration options)

## Contributing
(Brief contributing guide)

Be specific and accurate based on the actual code. Use proper Markdown formatting.""",

        'api': f"""You are a senior API documentation writer. Analyze this codebase for "{proj}" and generate comprehensive API Reference documentation.

FILE STRUCTURE:
{file_list}

CODEBASE:
{file_content}

Generate a detailed API_REFERENCE.md covering:
1. All API endpoints (method, path, description)
2. Request parameters (query params, body, headers)
3. Response format with examples
4. All exported functions/classes with:
   - Purpose
   - Parameters and types
   - Return values
   - Code examples
5. Error codes and messages
6. Authentication if present

Use proper Markdown with code blocks. Be precise and detailed.""",

        'architecture': f"""You are a software architect. Analyze this codebase for "{proj}" and document its architecture.

FILE STRUCTURE:
{file_list}

CODEBASE:
{file_content}

Generate a detailed ARCHITECTURE.md covering:

## Overview
(High-level description of the system)

## Architecture Pattern
(MVC, microservices, monolith, etc. — what you detect)

## Component Map
(All major components and their responsibilities)

## Data Flow
(How data moves through the system)

## Technology Stack
(With versions if detectable)

## Directory Structure
(Annotated directory tree)

## Key Design Decisions
(Patterns, conventions, notable choices found in code)

## Dependencies
(External libraries and why they're used)

## Deployment
(If detectable from config files)

Be specific about patterns and decisions you actually see in the code.""",

        'all': ""  # handled below
    }

    if doc_type == 'all':
        return build_prompt(files, 'readme', project_name)

    return prompts.get(doc_type, prompts['readme'])


# ─── AI streaming functions ───────────────────────────────────────────────────
def stream_claude(prompt: str, model: str, api_key: str) -> Generator:
    """Stream response from Anthropic Claude API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 8192,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are an expert technical documentation writer. Always respond with well-formatted Markdown.",
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as resp:
            if resp.status_code == 401:
                yield f"data: {json.dumps({'error': 'Invalid Anthropic API key.'})}\n\n"
                return
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'Claude API error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode('utf-8') if isinstance(line, bytes) else line
                if text.startswith('data: '):
                    data_str = text[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get('type') == 'content_block_delta':
                            token = data.get('delta', {}).get('text', '')
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        elif data.get('type') == 'message_stop':
                            yield f"data: {json.dumps({'done': True})}\n\n"
                    except Exception:
                        pass
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_ollama(prompt: str, model: str) -> Generator:
    """Stream response from local Ollama."""
    url = "http://localhost:11434/api/generate"
    try:
        with requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=180
        ) as resp:
            if resp.status_code == 404:
                err_msg = f'Model "{model}" not found. Run: ollama pull {model}'
                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                return
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'Ollama error: {resp.status_code}'})}\n\n"
                return
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get('response', '')
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if data.get('done'):
                        yield f"data: {json.dumps({'done': True})}\n\n"
    except requests.ConnectionError:
        yield f"data: {json.dumps({'error': 'Ollama is not running. Start it with: ollama serve'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_groq(prompt: str, model: str, api_key: str) -> Generator:
    """Stream response from Groq API (free tier)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert technical documentation writer. Always respond with well-formatted Markdown."},
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "max_tokens": 8192,
        "temperature": 0.3
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code == 401:
                yield f"data: {json.dumps({'error': 'Invalid Groq API key. Get one free at console.groq.com'})}\n\n"
                return
            if resp.status_code != 200:
                err = resp.text[:300]
                yield f"data: {json.dumps({'error': f'Groq API error {resp.status_code}: {err}'})}\n\n"
                return
            for line in resp.iter_lines():
                if line:
                    text = line.decode('utf-8') if isinstance(line, bytes) else line
                    if text.startswith('data: '):
                        data_str = text[6:]
                        if data_str == '[DONE]':
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                        try:
                            data = json.loads(data_str)
                            token = data['choices'][0]['delta'].get('content', '')
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception:
                            pass
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def stream_openai_compatible(prompt: str, model: str, api_key: str, base_url: str) -> Generator:
    """Stream from any OpenAI-compatible API (LM Studio, etc.)."""
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 8192,
    }
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'API error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                return
            for line in resp.iter_lines():
                if line:
                    text = line.decode('utf-8') if isinstance(line, bytes) else line
                    if text.startswith('data: '):
                        data_str = text[6:]
                        if data_str == '[DONE]':
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                        try:
                            data = json.loads(data_str)
                            token = data['choices'][0]['delta'].get('content', '')
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception:
                            pass
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/scan")
async def scan_repo(request: Request):
    """Scan and return file list without generating docs."""
    body = await request.json()
    source_type = body.get("source_type", "github")
    source = body.get("source", "").strip()

    if not source:
        return JSONResponse({"error": "No source provided"}, status_code=400)

    if source_type == "github":
        files = get_github_files(source)
    else:
        files = get_local_files(source)

    if '__error__' in files:
        return JSONResponse({"error": files['__error__']}, status_code=400)

    return JSONResponse({
        "file_count": len(files),
        "files": list(files.keys()),
        "total_chars": sum(len(v) for v in files.values())
    })


@app.post("/generate")
async def generate_docs(request: Request):
    """Main endpoint: generate documentation (SSE streaming)."""
    body = await request.json()

    source_type  = body.get("source_type", "github")
    source       = body.get("source", "").strip()
    doc_type     = body.get("doc_type", "readme")
    provider     = body.get("provider", "ollama")
    model        = body.get("model", "").strip()
    api_key      = body.get("api_key", "").strip()
    base_url     = body.get("base_url", "").strip()
    project_name = body.get("project_name", "").strip()

    if not source:
        async def err():
            yield f"data: {json.dumps({'error': 'No source provided'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # ── 1. Fetch files ────────────────────────────────────────────────────────
    if source_type == "github":
        files = get_github_files(source)
        if not project_name:
            parts = source.rstrip('/').split('/')
            project_name = parts[-1].replace('.git', '') if len(parts) >= 2 else source
    else:
        files = get_local_files(source)
        if not project_name:
            project_name = pathlib.Path(source).name

    if '__error__' in files:
        async def err():
            yield f"data: {json.dumps({'error': files['__error__']})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    if not files:
        async def err():
            yield f"data: {json.dumps({'error': 'No code files found in the source.'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    # ── 2. Build prompt ───────────────────────────────────────────────────────
    prompt = build_prompt(files, doc_type, project_name)

    # Emit file count info first
    info_msg = f"Found {len(files)} files. Generating {doc_type.upper()} documentation...\n\n"

    # ── 3. Choose provider ────────────────────────────────────────────────────
    def full_stream():
        yield f"data: {json.dumps({'info': info_msg})}\n\n"
        if provider == "ollama":
            yield from stream_ollama(prompt, model or "llama3.2")
        elif provider == "groq":
            yield from stream_groq(prompt, model or "llama-3.3-70b-versatile", api_key)
        elif provider == "claude":
            yield from stream_claude(prompt, model or "claude-opus-4-6", api_key)
        elif provider == "lmstudio":
            yield from stream_openai_compatible(
                prompt, model or "local-model",
                api_key, base_url or "http://localhost:1234/v1"
            )
        else:
            yield f"data: {json.dumps({'error': f'Unknown provider: {provider}'})}\n\n"

    return StreamingResponse(
        full_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/ollama/models")
async def list_ollama_models():
    """List available Ollama models."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            return JSONResponse({"models": models})
    except Exception:
        pass
    return JSONResponse({"models": [], "error": "Ollama not running"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print("\n" + "="*50)
    print("  LocalAIdoc - AI Documentation Generator")
    print(f"  Open: http://localhost:{port}")
    print("="*50 + "\n")
    uvicorn.run("main:app", host=host, port=port, reload=True)