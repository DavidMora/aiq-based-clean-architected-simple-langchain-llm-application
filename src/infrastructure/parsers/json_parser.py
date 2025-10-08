"""JSON output parser implementation."""
import json
import re
from typing import Optional

from src.domain.interfaces.llm_service import IOutputParser


class JSONOutputParser(IOutputParser):
    """Parser for JSON output from LLMs."""

    def __init__(self, strict: bool = False):
        """Initialize JSON parser.

        Args:
            strict: Whether to use strict JSON parsing
        """
        self.strict = strict

    def parse(self, text: str) -> dict:
        """Parse LLM output text as JSON.

        Args:
            text: Raw text output from LLM

        Returns:
            Parsed output as dictionary

        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # Try to find JSON object in text
        if not text.strip().startswith(('{', '[')):
            json_obj_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if json_obj_match:
                text = json_obj_match.group(1)

        try:
            return json.loads(text, strict=self.strict)
        except json.JSONDecodeError as e:
            if self.strict:
                raise ValueError(f"Failed to parse JSON: {e}")
            # Try more lenient parsing
            try:
                # Remove trailing commas
                cleaned = re.sub(r',\s*}', '}', text)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON output: {e}")


class LangChainOutputParser(IOutputParser):
    """Adapter for LangChain output parsers."""

    def __init__(self, parser):
        """Initialize with LangChain parser.

        Args:
            parser: LangChain output parser instance
        """
        self.parser = parser

    def parse(self, text: str) -> dict:
        """Parse using LangChain parser.

        Args:
            text: Raw text output from LLM

        Returns:
            Parsed output as dictionary
        """
        result = self.parser.parse(text)

        # Convert to dict if not already
        if isinstance(result, dict):
            return result
        elif hasattr(result, 'dict'):
            return result.dict()
        elif hasattr(result, '__dict__'):
            return result.__dict__
        else:
            return {"result": result}
