from .model_metadata import create_model_metadata

ALIBABA_CLOUD_MODELS_DETAILED = [
    create_model_metadata(
        provider="Alibaba Cloud", name="qwen3.6-plus", icon="AlibabaCloud", tool_calling=True, default=True
    ),
    create_model_metadata(
        provider="Alibaba Cloud", name="deepseek-v3.2", icon="AlibabaCloud", tool_calling=True, default=True
    ),
    create_model_metadata(
        provider="Alibaba Cloud", name="glm-5", icon="AlibabaCloud", tool_calling=True, default=True
    ),
    create_model_metadata(
        provider="Alibaba Cloud", name="MiniMax-M2.5", icon="AlibabaCloud", tool_calling=True, default=True
    ),
]

ALIBABA_CLOUD_MODELS = [
    metadata["name"]
    for metadata in ALIBABA_CLOUD_MODELS_DETAILED
    if not metadata.get("deprecated", False) and metadata.get("tool_calling", False)
]

DEFAULT_ALIBABA_CLOUD_API_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
