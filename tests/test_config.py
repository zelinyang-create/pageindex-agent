from pathlib import Path
from kb_agent.config import settings


def test_settings_paths_under_data_dir():
    assert settings.workspace_dir == settings.data_dir / "workspace"
    assert settings.catalog_path == settings.data_dir / "catalog" / "document_catalog.json"
    assert settings.domain_dict_path == settings.data_dir / "domain_dict_auto.txt"
    assert settings.index_dir == settings.data_dir / "indexes"
    assert settings.md_dir == settings.data_dir / "md"
    assert isinstance(settings.index_model, str)
