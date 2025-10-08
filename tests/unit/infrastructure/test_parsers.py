"""Tests for output parsers."""
import pytest
import json

from src.infrastructure.parsers.json_parser import JSONOutputParser, LangChainOutputParser


class TestJSONOutputParser:
    """Tests for JSONOutputParser."""

    def test_parse_simple_json(self, json_parser):
        """Test parsing simple JSON."""
        text = '{"result": 42, "status": "success"}'
        result = json_parser.parse(text)

        assert result == {"result": 42, "status": "success"}

    def test_parse_json_with_markdown(self, json_parser):
        """Test parsing JSON from markdown code block."""
        text = '''Here's the result:
```json
{
    "name": "test",
    "value": 123
}
```
'''
        result = json_parser.parse(text)

        assert result == {"name": "test", "value": 123}

    def test_parse_json_without_markdown_label(self, json_parser):
        """Test parsing JSON from code block without json label."""
        text = '''```
{"key": "value"}
```'''
        result = json_parser.parse(text)

        assert result == {"key": "value"}

    def test_parse_json_embedded_in_text(self, json_parser):
        """Test parsing JSON embedded in text."""
        text = 'The answer is {"result": 4, "operation": "addition"} based on the calculation.'
        result = json_parser.parse(text)

        assert result == {"result": 4, "operation": "addition"}

    def test_parse_json_array(self, json_parser):
        """Test parsing JSON array."""
        text = '[1, 2, 3, 4, 5]'
        result = json_parser.parse(text)

        assert result == [1, 2, 3, 4, 5]

    def test_parse_json_with_trailing_comma(self, json_parser):
        """Test parsing JSON with trailing comma (lenient mode)."""
        text = '{"key": "value",}'
        result = json_parser.parse(text)

        assert result == {"key": "value"}

    def test_parse_invalid_json_raises_error(self, json_parser):
        """Test that invalid JSON raises ValueError."""
        text = 'This is not JSON at all'

        with pytest.raises(ValueError) as exc_info:
            json_parser.parse(text)

        assert "Failed to parse JSON output" in str(exc_info.value)

    def test_parse_empty_string_raises_error(self, json_parser):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            json_parser.parse("")

    def test_strict_mode_with_trailing_comma(self):
        """Test strict mode rejects trailing commas."""
        parser = JSONOutputParser(strict=True)
        text = '{"key": "value",}'

        with pytest.raises(ValueError):
            parser.parse(text)

    def test_parse_nested_json(self, json_parser):
        """Test parsing nested JSON."""
        text = '''
{
    "user": {
        "name": "John",
        "age": 30
    },
    "items": [1, 2, 3]
}
'''
        result = json_parser.parse(text)

        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30
        assert result["items"] == [1, 2, 3]


class TestLangChainOutputParser:
    """Tests for LangChainOutputParser."""

    def test_parse_with_dict_result(self):
        """Test parsing when LangChain parser returns dict."""
        mock_parser = lambda text: {"parsed": True}
        mock_parser.parse = lambda text: {"result": "success"}

        parser = LangChainOutputParser(mock_parser)
        result = parser.parse("test input")

        assert result == {"result": "success"}

    def test_parse_with_object_with_dict_method(self):
        """Test parsing when result has dict() method."""
        class MockResult:
            def dict(self):
                return {"converted": True}

        mock_parser = lambda text: MockResult()
        mock_parser.parse = lambda text: MockResult()

        parser = LangChainOutputParser(mock_parser)
        result = parser.parse("test input")

        assert result == {"converted": True}

    def test_parse_with_object_with_dict_attribute(self):
        """Test parsing when result has __dict__ attribute."""
        class MockResult:
            def __init__(self):
                self.key = "value"
                self.number = 42

        mock_parser = lambda text: MockResult()
        mock_parser.parse = lambda text: MockResult()

        parser = LangChainOutputParser(mock_parser)
        result = parser.parse("test input")

        assert "key" in result
        assert result["key"] == "value"

    def test_parse_with_primitive_result(self):
        """Test parsing when result is a primitive."""
        mock_parser = lambda text: "simple string"
        mock_parser.parse = lambda text: "simple string"

        parser = LangChainOutputParser(mock_parser)
        result = parser.parse("test input")

        assert result == {"result": "simple string"}
