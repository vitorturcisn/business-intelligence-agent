
"""Callbacks de execução para o Business Intelligence Agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class AgentLoggingCallbackHandler(BaseCallbackHandler):
    """Registra eventos de execução sem gravar dados sensíveis."""

    def __init__(
        self,
        logger_name: str = "business_agent.execution",
    ) -> None:
        self.logger = logging.getLogger(logger_name)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        self.logger.info("tool.start name=%s", tool_name)

    def on_tool_end(
        self,
        output: Any,
        **kwargs: Any,
    ) -> None:
        self.logger.info(
            "tool.end output_type=%s",
            type(output).__name__,
        )

    def on_tool_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        self.logger.warning(
            "tool.error error_type=%s",
            type(error).__name__,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        model_name = serialized.get("name", "unknown")
        self.logger.info("llm.start name=%s", model_name)

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any,
    ) -> None:
        self.logger.info("llm.end")

    def on_chain_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        self.logger.warning(
            "chain.error error_type=%s",
            type(error).__name__,
        )


def configure_logging(level: int = logging.INFO) -> None:
    """Configura o formato básico dos logs da aplicação."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
