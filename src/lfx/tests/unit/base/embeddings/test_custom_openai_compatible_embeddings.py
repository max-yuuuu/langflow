from lfx.base.models import model_utils
from lfx.base.models.unified_models import model_catalog


def test_fetch_live_custom_openai_compatible_models_filters_embeddings(monkeypatch):
    monkeypatch.setattr(
        model_utils,
        "get_provider_variable_value",
        lambda user_id, variable_key: {  # noqa: ARG005
            "CUSTOM_OPENAI_BASE_URL": "https://example.test/v1",
            "CUSTOM_OPENAI_API_KEY": "secret",
        }.get(variable_key),
    )
    monkeypatch.setattr(
        model_utils,
        "fetch_openai_compatible_model_names",
        lambda base_url, api_key: [  # noqa: ARG005
            "qwen-max",
            "text-embedding-v3",
            "bge-m3-embedding",
        ],
    )

    embedding_models = model_utils.fetch_live_custom_openai_compatible_models("user-id", "embeddings")
    llm_models = model_utils.fetch_live_custom_openai_compatible_models("user-id", "llm")

    assert [model["name"] for model in embedding_models] == ["text-embedding-v3", "bge-m3-embedding"]
    assert all(model["model_type"] == "embeddings" for model in embedding_models)
    assert all(model["tool_calling"] is False for model in embedding_models)

    assert [model["name"] for model in llm_models] == ["qwen-max"]
    assert all(model["model_type"] == "llm" for model in llm_models)
    assert all(model["tool_calling"] is True for model in llm_models)


def test_get_embedding_model_options_supports_custom_openai_compatible(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "get_unified_models_detailed",
        lambda **kwargs: [  # noqa: ARG005
            {
                "provider": "Custom OpenAI Compatible",
                "icon": "Bot",
                "models": [
                    {
                        "model_name": "text-embedding-v3",
                        "metadata": {"default": True, "model_type": "embeddings"},
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(model_catalog, "replace_with_live_models", lambda *args, **kwargs: args[0])  # noqa: ARG005

    async def mock_get_model_status(user_id):  # noqa: ARG001
        return set(), set()

    async def mock_fetch_enabled_providers(user_id):  # noqa: ARG001
        return {"Custom OpenAI Compatible"}

    monkeypatch.setattr(model_catalog, "_get_model_status", mock_get_model_status)
    monkeypatch.setattr(model_catalog, "_fetch_enabled_providers_for_user", mock_fetch_enabled_providers)

    options = model_catalog.get_embedding_model_options("user-id")

    assert len(options) == 1
    option = options[0]
    assert option["provider"] == "Custom OpenAI Compatible"
    assert option["metadata"]["embedding_class"] == "OpenAIEmbeddings"
    assert option["metadata"]["param_mapping"]["api_base"] == "base_url"
