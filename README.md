# LocalAIdoc

Lokal va bepul ishlaydigan AI hujjat generatori. [superdocs.cloud](https://www.superdocs.cloud/) ga o'xshash, lekin internetga yuborilmaydi.

## Imkoniyatlar

- **GitHub URL** yoki **lokal papka** dan kod o'qiydi
- **3 turdagi hujjat** yaratadi: README, API Reference, Architecture
- **Real-time streaming** — natija harfma-harf chiqadi
- **Preview + Raw** ko'rinish
- **Copy & Download** (.md fayl sifatida)
- **3 ta bepul AI provider** qo'llab-quvvatlanadi

## AI Providerlar

| Provider | Narx | Tavsif |
|---|---|---|
| **Ollama** | Bepul, to'liq lokal | Internetga ulanmaydi |
| **Groq** | Bepul API | Eng tez, `console.groq.com` da key oling |
| **LM Studio** | Bepul, lokal | GUI orqali model yuklash |

## O'rnatish

```bash
# 1. Dependencylarni o'rnating
pip install -r requirements.txt

# 2. Serverni ishga tushiring
py main.py
```

## Ishlatish

```
http://localhost:8000
```

1. **Source** — GitHub URL yoki lokal papka yo'lini kiriting
2. **Doc Type** — README / API Ref / Architecture tanlang
3. **AI Provider** — Ollama, Groq yoki LM Studio tanlang
4. **Generate** tugmasini bosing

## Groq (eng tez yo'l)

```
1. https://console.groq.com ga kiring
2. Bepul account oching va API key oling
3. UI da "Groq" tanlang, keyni kiriting
4. Generate bosing
```

## Ollama (to'liq lokal)

```bash
# 1. https://ollama.com dan o'rnating
# 2. Model yuklab oling
ollama pull llama3.2

# 3. Ishga tushiring
ollama serve
```

## Loyiha strukturasi

```
localAIdoc/
├── main.py              # FastAPI backend
├── requirements.txt     # Python kutubxonalar
├── README.md
└── templates/
    └── index.html       # Web UI
```

## Texnologiyalar

- **Python 3.12** + **FastAPI** — backend
- **Server-Sent Events (SSE)** — real-time streaming
- **Jinja2** — HTML shablonlar
- **marked.js** — Markdown render
