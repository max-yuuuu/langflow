from typing import Any

import requests

from lfx.base.models.alibaba_cloud_constants import (
    ALIBABA_CLOUD_MODELS,
    DEFAULT_ALIBABA_CLOUD_API_URL,
)
from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.io import BoolInput, DropdownInput, IntInput, MessageTextInput, SecretStrInput, SliderInput
from lfx.log.logger import logger
from lfx.schema.dotdict import dotdict


class AlibabaCloudModelComponent(LCModelComponent):
    display_name = "Alibaba Cloud"
    description = "Generate text using Alibaba Cloud Bailian's Anthropic-compatible API."
    icon = "AlibabaCloud"
    name = "AlibabaCloudModel"

    inputs = [
        *LCModelComponent.get_base_inputs(),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            value=4096,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=ALIBABA_CLOUD_MODELS,
            refresh_button=True,
            value=ALIBABA_CLOUD_MODELS[0],
            combobox=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="Alibaba Cloud API Key",
            info="Your Alibaba Cloud Bailian API key.",
            value=None,
            required=True,
            real_time_refresh=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Run inference with this temperature. Must be in the closed interval [0.0, 1.0].",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Alibaba Cloud API URL",
            info="Endpoint of the Alibaba Cloud Bailian Anthropic-compatible API.",
            value=DEFAULT_ALIBABA_CLOUD_API_URL,
            real_time_refresh=True,
            advanced=True,
        ),
        BoolInput(
            name="tool_model_enabled",
            display_name="Enable Tool Models",
            info=(
                "Select if you want to use models that can work with tools. If yes, only those models will be shown."
            ),
            advanced=False,
            value=False,
            real_time_refresh=True,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            from langchain_anthropic.chat_models import ChatAnthropic
        except ImportError as e:
            msg = "langchain_anthropic is not installed. Please install it with `pip install langchain_anthropic`."
            raise ImportError(msg) from e
        try:
            max_tokens_value = getattr(self, "max_tokens", "")
            max_tokens_value = 4096 if max_tokens_value == "" else int(max_tokens_value)
            output = ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=self.api_key,
                max_tokens=max_tokens_value,
                temperature=self.temperature,
                anthropic_api_url=self.base_url or DEFAULT_ALIBABA_CLOUD_API_URL,
                streaming=self.stream,
                stream_usage=True,
            )
        except Exception as e:
            msg = "Could not connect to Alibaba Cloud API."
            raise ValueError(msg) from e

        return output

    def get_models(self, *, tool_model_enabled: bool | None = None) -> list[str]:
        # For Alibaba Cloud, use static model list since it may not support
        # the Anthropic models listing API
        model_ids = list(ALIBABA_CLOUD_MODELS)

        if tool_model_enabled:
            # All listed models support tool calling
            return [m for m in model_ids]

        return model_ids

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if "base_url" in build_config and build_config["base_url"]["value"] is None:
            build_config["base_url"]["value"] = DEFAULT_ALIBABA_CLOUD_API_URL
            self.base_url = DEFAULT_ALIBABA_CLOUD_API_URL
        if field_name in {"base_url", "model_name", "tool_model_enabled", "api_key"} and field_value:
            try:
                if not self.api_key or len(self.api_key) == 0:
                    ids = ALIBABA_CLOUD_MODELS
                else:
                    try:
                        ids = self.get_models(tool_model_enabled=self.tool_model_enabled)
                    except (ImportError, ValueError, requests.exceptions.RequestException) as e:
                        logger.exception(f"Error getting model names: {e}")
                        ids = ALIBABA_CLOUD_MODELS
                build_config.setdefault("model_name", {})
                build_config["model_name"]["options"] = ids
                build_config["model_name"].setdefault("value", ids[0])
                build_config["model_name"]["combobox"] = True
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config
