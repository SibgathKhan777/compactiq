"""
stack_suggest.py - Keyword-based project -> package-stack suggestion for the
Dependency Assistant chat.

When a chat message describes a project instead of pasting actual
`pip install` code (e.g. "I'm building a web scraper"), there's nothing for
validate.py to check. This module maps a small set of project-type keywords to
a typical package stack -- but ONLY packages present in this repo's trained
catalog (data.json), so every suggestion can immediately be run through the
real validate_install_code() pipeline (ML + live PyPI + dependency conflicts)
instead of being an unverified guess. This is intentionally NOT a general
"recommend packages for any project" feature -- it's a small curated map, and
it says so when nothing matches rather than inventing a stack.
"""

# Each entry: (keywords to match, packages from the trained catalog, human label)
STACK_MAP = [
    (["web scraper", "scraping", "scrape site", "crawler"], ["requests", "beautifulsoup4"], "web scraping"),
    (["machine learning", " ml ", "scikit", "sklearn", "data science model"], ["numpy", "pandas", "scikit-learn"], "machine learning"),
    (["rest api", "api server", "backend api", "fastapi"], ["fastapi", "uvicorn", "pydantic"], "REST API"),
    (["flask app", "web app", "website backend"], ["flask"], "Flask web app"),
    (["s3", "aws", "cloud storage bucket"], ["boto3", "botocore"], "AWS/S3"),
    (["database", "sql orm", "db migrations", " orm "], ["sqlalchemy", "alembic"], "SQL database + migrations"),
    (["llm", "langchain", "chatbot", "ai agent", "gpt"], ["langchain", "openai"], "LLM/agent app"),
    (["kubernetes", "k8s", "container orchestration"], ["kubernetes"], "Kubernetes automation"),
    (["data pipeline", "data engineering", "etl"], ["pandas", "pyarrow", "fsspec"], "data pipeline/ETL"),
    (["unit test", "test suite", "testing framework"], ["pytest"], "test suite"),
    (["cli tool", "command line tool", "command-line tool"], ["click", "typer", "rich"], "CLI tool"),
    (["google cloud", "gcp", "bigquery"], ["google-cloud-bigquery", "google-cloud-storage", "google-auth"], "Google Cloud"),
]


def suggest_stack(message):
    """
    Returns (label, packages) for the first matching category, or (None, [])
    if nothing in the message matches a known project type.
    """
    lowered = f" {message.lower()} "
    for keywords, packages, label in STACK_MAP:
        if any(kw in lowered for kw in keywords):
            return label, packages
    return None, []


def known_categories():
    """Human-readable list of project types this can suggest a stack for."""
    return [label for _, _, label in STACK_MAP]
