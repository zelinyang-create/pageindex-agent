import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_DIR", "data"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    index_model: str = field(default_factory=lambda: os.getenv("PAGEINDEX_INDEX_MODEL", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", ""))

    @property
    def md_dir(self) -> Path:
        return self.data_dir / "md"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog" / "document_catalog.json"

    @property
    def domain_dict_path(self) -> Path:
        return self.data_dir / "domain_dict_auto.txt"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"


settings = Settings()
