import logfire
import time
from app.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  #all-mpnet-base-v2

_active_model = None
_model_type = str | None #"gemini" or "fallback"


# --------Model Initialization --------

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GOOGLE_API_KEY,
        )
        # Test the model
        model.embed_query("Test")
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim)")
        return model
    except Exception as e:
        logfire.error(f"Error occurred while probing Gemini: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback model: all-mpnet-base-v2 (768-dim)")
    return SentenceTransformer("all-mpnet-base-v2")



def _init():
    global _active_model, _model_type
    
    if _active_model is not None:
        return  # Already initialized
    
    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"



# --- Public helpers -----------------

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# -------Batch embedding with retry -----------------

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ["429","rate","rate limit", "quota", "resource_exhausted"])  # Add rate limit error messages if known
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit - retrying in {wait} seconds (attempt {attempt+1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}.")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 retries.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()  # SentenceTransformer returns numpy array

#-----Public API (same signatures as before)-----------

def embed_query(query:str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("Embedding batch", model = _model_type, start = i, size=len(batch)):
            embeddings = _embed_batch(batch)
            all_embeddings.extend(embeddings)

    return all_embeddings