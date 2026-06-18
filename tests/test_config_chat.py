from kb_agent.config import settings

def test_chat_settings_exist():
    assert hasattr(settings, "openrouter_api_key")
    assert settings.openrouter_base_url  # 有默认值
    assert hasattr(settings, "chat_model")
