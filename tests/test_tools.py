"""Unit tests for GoodMem CrewAI tools.

HTTP calls are mocked; no live GoodMem server is required.

Run with:
    uv run pytest -m "not e2e"
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_goodmem import (
    GoodMemCreateMemoryTool,
    GoodMemCreateSpaceTool,
    GoodMemDeleteMemoryTool,
    GoodMemDeleteSpaceTool,
    GoodMemGetMemoryTool,
    GoodMemGetSpaceTool,
    GoodMemListEmbeddersTool,
    GoodMemListMemoriesTool,
    GoodMemListSpacesTool,
    GoodMemRetrieveMemoriesTool,
    GoodMemUpdateSpaceTool,
)
from crewai_goodmem.tools import _mime_from_extension


BASE_URL = "https://test.goodmem.ai"
API_KEY = "gm_test_key_123"

_MODULE = "crewai_goodmem.tools"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure env vars don't leak between tests."""
    monkeypatch.delenv("GOODMEM_BASE_URL", raising=False)
    monkeypatch.delenv("GOODMEM_API_KEY", raising=False)


def _tool_kwargs():
    return dict(base_url=BASE_URL, api_key=API_KEY, verify_ssl=False)


def _make_response(
    json_data=None,
    text=None,
    content=None,
    headers=None,
    status_code=200,
    raise_for_status=None,
):
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    if content is not None:
        resp.content = content
    if text is not None:
        resp.text = text
    else:
        resp.text = json.dumps(json_data) if json_data else ""
    resp.headers = headers or {}
    if raise_for_status:
        resp.raise_for_status.side_effect = raise_for_status
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Initialization / Config resolution
# ---------------------------------------------------------------------------


class TestToolInitialization:
    """All tools should instantiate with explicit args or env vars."""

    TOOL_CLASSES = [
        GoodMemCreateMemoryTool,
        GoodMemCreateSpaceTool,
        GoodMemDeleteMemoryTool,
        GoodMemDeleteSpaceTool,
        GoodMemGetMemoryTool,
        GoodMemGetSpaceTool,
        GoodMemListEmbeddersTool,
        GoodMemListMemoriesTool,
        GoodMemListSpacesTool,
        GoodMemRetrieveMemoriesTool,
        GoodMemUpdateSpaceTool,
    ]

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_init_with_explicit_args(self, tool_class):
        tool = tool_class(**_tool_kwargs())
        assert isinstance(tool, tool_class)
        assert tool.base_url == BASE_URL
        assert tool.api_key == API_KEY
        assert tool.verify_ssl is False

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_init_with_env_vars(self, tool_class, monkeypatch):
        monkeypatch.setenv("GOODMEM_BASE_URL", BASE_URL)
        monkeypatch.setenv("GOODMEM_API_KEY", API_KEY)
        tool = tool_class()
        assert isinstance(tool, tool_class)

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_missing_config_returns_error_json(self, tool_class):
        """Tools should return a JSON error (not raise) when config is missing."""
        tool = tool_class()

        if tool_class == GoodMemCreateSpaceTool:
            result = tool._run(name="test", embedder_id="emb-1")
        elif tool_class == GoodMemUpdateSpaceTool:
            result = tool._run(space_id="sp-1", name="new-name")
        elif tool_class == GoodMemDeleteSpaceTool:
            result = tool._run(space_id="sp-1")
        elif tool_class == GoodMemGetSpaceTool:
            result = tool._run(space_id="sp-1")
        elif tool_class == GoodMemCreateMemoryTool:
            result = tool._run(space_id="sp-1", text_content="hello")
        elif tool_class == GoodMemListMemoriesTool:
            result = tool._run(space_id="sp-1")
        elif tool_class == GoodMemRetrieveMemoriesTool:
            result = tool._run(query="test", space_ids=["sp-1"])
        elif tool_class == GoodMemGetMemoryTool:
            result = tool._run(memory_id="mem-1")
        elif tool_class == GoodMemDeleteMemoryTool:
            result = tool._run(memory_id="mem-1")
        elif tool_class in (GoodMemListSpacesTool, GoodMemListEmbeddersTool):
            result = tool._run()
        else:
            pytest.fail(f"Unknown tool class: {tool_class}")

        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data
        assert "GOODMEM" in data["error"]


class TestBaseUrlNormalization:
    def test_trailing_slash_stripped(self):
        """base_url with trailing slash should be normalized (no double slashes)."""
        with patch(f"{_MODULE}.requests.get") as mock_get:
            mock_get.return_value = _make_response(json_data={"spaces": []})
            tool = GoodMemListSpacesTool(
                base_url="https://test.goodmem.ai/", api_key=API_KEY
            )
            tool._run()
            url = mock_get.call_args[0][0]
            assert url == "https://test.goodmem.ai/v1/spaces"


# ---------------------------------------------------------------------------
# MIME type helper
# ---------------------------------------------------------------------------


class TestMimeType:
    def test_known_extensions(self):
        assert _mime_from_extension("pdf") == "application/pdf"
        assert _mime_from_extension("PNG") == "image/png"
        assert _mime_from_extension(".jpg") == "image/jpeg"
        assert _mime_from_extension("txt") == "text/plain"
        assert _mime_from_extension("md") == "text/markdown"

    def test_unknown_extension(self):
        assert _mime_from_extension("xyz") is None
        assert _mime_from_extension("") is None


# ---------------------------------------------------------------------------
# GoodMemListEmbeddersTool
# ---------------------------------------------------------------------------


class TestListEmbeddersTool:
    @patch(f"{_MODULE}.requests.get")
    def test_list_embedders_dict_response(self, mock_get):
        mock_get.return_value = _make_response(
            json_data={
                "embedders": [
                    {
                        "embedderId": "emb-1",
                        "displayName": "Test Embedder",
                        "modelIdentifier": "model-v1",
                    }
                ]
            }
        )
        tool = GoodMemListEmbeddersTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["success"] is True
        assert result["totalEmbedders"] == 1
        assert result["embedders"][0]["embedderId"] == "emb-1"
        assert result["embedders"][0]["displayName"] == "Test Embedder"
        assert result["embedders"][0]["modelIdentifier"] == "model-v1"
        # GETs should not include Content-Type
        headers = mock_get.call_args[1]["headers"]
        assert "Content-Type" not in headers

    @patch(f"{_MODULE}.requests.get")
    def test_list_embedders_list_response(self, mock_get):
        mock_get.return_value = _make_response(
            json_data=[{"id": "emb-2", "name": "Fallback Name", "model": "m2"}]
        )
        tool = GoodMemListEmbeddersTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["embedders"][0]["embedderId"] == "emb-2"
        assert result["embedders"][0]["displayName"] == "Fallback Name"
        assert result["embedders"][0]["modelIdentifier"] == "m2"

    @patch(f"{_MODULE}.requests.get")
    def test_list_embedders_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        tool = GoodMemListEmbeddersTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["success"] is False
        assert "Failed to list embedders" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemListSpacesTool
# ---------------------------------------------------------------------------


class TestListSpacesTool:
    @patch(f"{_MODULE}.requests.get")
    def test_list_spaces_dict_response(self, mock_get):
        mock_get.return_value = _make_response(
            json_data={
                "spaces": [
                    {
                        "spaceId": "sp-1",
                        "name": "My Space",
                        "spaceEmbedders": [{"embedderId": "emb-1"}],
                    },
                    {
                        "spaceId": "sp-2",
                        "name": "Other Space",
                        "spaceEmbedders": [],
                    },
                ]
            }
        )
        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["success"] is True
        assert result["totalSpaces"] == 2
        assert result["spaces"][0]["spaceId"] == "sp-1"
        assert result["spaces"][0]["spaceEmbedders"] == [{"embedderId": "emb-1"}]
        assert result["spaces"][1]["spaceEmbedders"] == []
        headers = mock_get.call_args[1]["headers"]
        assert "Content-Type" not in headers

    @patch(f"{_MODULE}.requests.get")
    def test_list_spaces_empty(self, mock_get):
        mock_get.return_value = _make_response(json_data={"spaces": []})
        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["spaces"] == []
        assert result["totalSpaces"] == 0

    @patch(f"{_MODULE}.requests.get")
    def test_list_spaces_no_embedders_field_defaults_to_empty(self, mock_get):
        mock_get.return_value = _make_response(
            json_data={"spaces": [{"spaceId": "sp-1", "name": "My Space"}]}
        )
        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["spaces"][0]["spaceEmbedders"] == []

    @patch(f"{_MODULE}.requests.get")
    def test_list_spaces_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        http_err = requests.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err
        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())
        assert result["success"] is False
        assert "Failed to list spaces" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemGetSpaceTool
# ---------------------------------------------------------------------------


class TestGetSpaceTool:
    @patch(f"{_MODULE}.requests.get")
    def test_get_space_success(self, mock_get):
        mock_get.return_value = _make_response(
            json_data={
                "spaceId": "sp-1",
                "name": "My Space",
                "spaceEmbedders": [{"embedderId": "emb-1"}],
            }
        )
        tool = GoodMemGetSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["success"] is True
        assert result["space"]["spaceId"] == "sp-1"
        assert result["space"]["name"] == "My Space"
        url = mock_get.call_args[0][0]
        assert url.endswith("/v1/spaces/sp-1")
        headers = mock_get.call_args[1]["headers"]
        assert "Content-Type" not in headers

    @patch(f"{_MODULE}.requests.get")
    def test_get_space_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        http_err = requests.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err
        tool = GoodMemGetSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="nonexistent"))
        assert result["success"] is False
        assert "Failed to get space" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemCreateSpaceTool
# ---------------------------------------------------------------------------


class TestCreateSpaceTool:
    @patch(f"{_MODULE}.requests.post")
    @patch(f"{_MODULE}.requests.get")
    def test_create_space_new(self, mock_get, mock_post):
        mock_get.return_value = _make_response(json_data={"spaces": []})
        mock_post.return_value = _make_response(
            json_data={"spaceId": "sp-new", "name": "test-space"}
        )
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="test-space", embedder_id="emb-1"))
        assert result["success"] is True
        assert result["spaceId"] == "sp-new"
        assert result["reused"] is False
        post_headers = mock_post.call_args[1]["headers"]
        assert post_headers["Content-Type"] == "application/json"

    @patch(f"{_MODULE}.requests.post")
    @patch(f"{_MODULE}.requests.get")
    def test_create_space_reuses_existing(self, mock_get, mock_post):
        mock_get.return_value = _make_response(
            json_data={"spaces": [{"spaceId": "sp-existing", "name": "my-space"}]}
        )
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="my-space", embedder_id="emb-1"))
        assert result["success"] is True
        assert result["spaceId"] == "sp-existing"
        assert result["reused"] is True
        mock_post.assert_not_called()

    @patch(f"{_MODULE}.requests.get")
    def test_create_space_reuse_returns_actual_embedder(self, mock_get):
        """Reused space should expose the embedder actually attached to the
        space, not the one the caller passed."""
        mock_get.return_value = _make_response(
            json_data={
                "spaces": [
                    {
                        "spaceId": "sp-existing",
                        "name": "my-space",
                        "spaceEmbedders": [{"embedderId": "actual-emb-from-server"}],
                    }
                ]
            }
        )
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="my-space", embedder_id="caller-emb"))
        assert result["success"] is True
        assert result["reused"] is True
        assert result["embedderId"] == "actual-emb-from-server"

    @patch(f"{_MODULE}.requests.post")
    @patch(f"{_MODULE}.requests.get")
    def test_create_space_list_fails_still_creates(self, mock_get, mock_post):
        """Non-auth errors from list_spaces should fall through to create."""
        mock_get.side_effect = requests.ConnectionError("connection refused")
        mock_post.return_value = _make_response(
            json_data={"spaceId": "sp-new", "name": "test-space"}
        )
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="test-space", embedder_id="emb-1"))
        assert result["success"] is True
        assert result["reused"] is False

    @patch(f"{_MODULE}.requests.post")
    @patch(f"{_MODULE}.requests.get")
    def test_create_space_auth_failure_on_list(self, mock_get, mock_post):
        """Auth errors listing spaces should be returned, not silently swallowed."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        http_err = requests.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="test", embedder_id="emb-1"))
        assert result["success"] is False
        assert "Authentication failed" in result["error"]
        mock_post.assert_not_called()

    @patch(f"{_MODULE}.requests.post")
    @patch(f"{_MODULE}.requests.get")
    def test_create_space_post_error(self, mock_get, mock_post):
        mock_get.return_value = _make_response(json_data={"spaces": []})
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        http_err = requests.HTTPError(response=mock_resp)
        mock_post.side_effect = http_err
        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="fail", embedder_id="emb-1"))
        assert result["success"] is False
        assert "Failed to create space" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemUpdateSpaceTool
# ---------------------------------------------------------------------------


class TestUpdateSpaceTool:
    @patch(f"{_MODULE}.requests.put")
    def test_update_name(self, mock_put):
        mock_put.return_value = _make_response(
            json_data={"spaceId": "sp-1", "name": "renamed"}
        )
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1", name="renamed"))
        assert result["success"] is True
        assert result["space"]["name"] == "renamed"
        body = mock_put.call_args[1]["json"]
        assert body == {"name": "renamed"}
        url = mock_put.call_args[0][0]
        assert url.endswith("/v1/spaces/sp-1")

    @patch(f"{_MODULE}.requests.put")
    def test_update_public_read(self, mock_put):
        mock_put.return_value = _make_response(
            json_data={"spaceId": "sp-1", "publicRead": True}
        )
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        tool._run(space_id="sp-1", public_read=True)
        body = mock_put.call_args[1]["json"]
        assert body == {"publicRead": True}

    @patch(f"{_MODULE}.requests.put")
    def test_update_replace_labels(self, mock_put):
        mock_put.return_value = _make_response(json_data={"spaceId": "sp-1"})
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        tool._run(space_id="sp-1", replace_labels_json='{"env": "prod"}')
        body = mock_put.call_args[1]["json"]
        assert body == {"replaceLabels": {"env": "prod"}}

    @patch(f"{_MODULE}.requests.put")
    def test_update_merge_labels(self, mock_put):
        mock_put.return_value = _make_response(json_data={"spaceId": "sp-1"})
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        tool._run(space_id="sp-1", merge_labels_json='{"team": "ml"}')
        body = mock_put.call_args[1]["json"]
        assert body == {"mergeLabels": {"team": "ml"}}

    @patch(f"{_MODULE}.requests.put")
    def test_update_both_labels_returns_error(self, mock_put):
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                space_id="sp-1",
                replace_labels_json='{"a": "b"}',
                merge_labels_json='{"c": "d"}',
            )
        )
        assert result["success"] is False
        assert "Cannot use both" in result["error"]
        mock_put.assert_not_called()

    @patch(f"{_MODULE}.requests.put")
    def test_update_invalid_replace_labels_json(self, mock_put):
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                space_id="sp-1",
                replace_labels_json="not-json",
            )
        )
        assert result["success"] is False
        assert "replace_labels_json is not valid JSON" in result["error"]
        mock_put.assert_not_called()

    @patch(f"{_MODULE}.requests.put")
    def test_update_error_propagates(self, mock_put):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_put.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemUpdateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="nonexistent", name="x"))
        assert result["success"] is False
        assert "Failed to update space" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemDeleteSpaceTool
# ---------------------------------------------------------------------------


class TestDeleteSpaceTool:
    @patch(f"{_MODULE}.requests.delete")
    def test_delete_success(self, mock_delete):
        mock_delete.return_value = _make_response(json_data={})
        tool = GoodMemDeleteSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["success"] is True
        assert result["spaceId"] == "sp-1"
        headers = mock_delete.call_args[1]["headers"]
        assert "Content-Type" not in headers
        url = mock_delete.call_args[0][0]
        assert url.endswith("/v1/spaces/sp-1")

    @patch(f"{_MODULE}.requests.delete")
    def test_delete_not_found(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_delete.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemDeleteSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="nonexistent"))
        assert result["success"] is False
        assert "Failed to delete space" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemCreateMemoryTool
# ---------------------------------------------------------------------------


class TestCreateMemoryTool:
    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_text(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={
                "memoryId": "mem-1",
                "spaceId": "sp-1",
                "processingStatus": "PENDING",
            }
        )
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1", text_content="Hello world"))
        assert result["success"] is True
        assert result["memoryId"] == "mem-1"
        assert result["contentType"] == "text/plain"
        assert result["status"] == "PENDING"
        body = mock_post.call_args[1]["json"]
        assert body["spaceId"] == "sp-1"
        assert body["contentType"] == "text/plain"
        assert body["originalContent"] == "Hello world"

    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_text_file(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={
                "memoryId": "mem-f",
                "spaceId": "sp-1",
                "processingStatus": "PENDING",
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("file content here")
            tmp_path = f.name
        try:
            tool = GoodMemCreateMemoryTool(**_tool_kwargs())
            result = json.loads(tool._run(space_id="sp-1", file_path=tmp_path))
            assert result["success"] is True
            assert result["contentType"] == "text/plain"
            body = mock_post.call_args[1]["json"]
            assert body["originalContent"] == "file content here"
            assert "originalContentB64" not in body
        finally:
            os.unlink(tmp_path)

    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_binary_file(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={
                "memoryId": "mem-pdf",
                "spaceId": "sp-1",
                "processingStatus": "PENDING",
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-fake-content")
            tmp_path = f.name
        try:
            tool = GoodMemCreateMemoryTool(**_tool_kwargs())
            result = json.loads(tool._run(space_id="sp-1", file_path=tmp_path))
            assert result["success"] is True
            assert result["contentType"] == "application/pdf"
            body = mock_post.call_args[1]["json"]
            assert "originalContentB64" in body
            assert "originalContent" not in body
            decoded = base64.b64decode(body["originalContentB64"])
            assert decoded == b"%PDF-fake-content"
        finally:
            os.unlink(tmp_path)

    def test_create_memory_no_content(self):
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["success"] is False
        assert "No content provided" in result["error"]

    def test_create_memory_bad_file_path(self):
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(
            tool._run(space_id="sp-1", file_path="/nonexistent/file.pdf")
        )
        assert result["success"] is False
        assert "Cannot read file" in result["error"]

    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_with_metadata(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={
                "memoryId": "mem-m",
                "spaceId": "sp-1",
                "processingStatus": "PENDING",
            }
        )
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                space_id="sp-1",
                text_content="test",
                metadata={"category": "feat"},
            )
        )
        assert result["success"] is True
        body = mock_post.call_args[1]["json"]
        assert body["metadata"] == {"category": "feat"}

    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_file_takes_priority(self, mock_post):
        mock_post.return_value = _make_response(
            json_data={
                "memoryId": "mem-md",
                "spaceId": "sp-1",
                "processingStatus": "PENDING",
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Markdown content")
            tmp_path = f.name
        try:
            tool = GoodMemCreateMemoryTool(**_tool_kwargs())
            result = json.loads(
                tool._run(
                    space_id="sp-1",
                    text_content="ignored text",
                    file_path=tmp_path,
                )
            )
            assert result["contentType"] == "text/markdown"
            body = mock_post.call_args[1]["json"]
            assert body["originalContent"] == "# Markdown content"
        finally:
            os.unlink(tmp_path)

    @patch(f"{_MODULE}.requests.post")
    def test_create_memory_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"
        mock_post.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1", text_content="hi"))
        assert result["success"] is False
        assert "Failed to create memory" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemListMemoriesTool
# ---------------------------------------------------------------------------


class TestListMemoriesTool:
    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_dict_response(self, mock_get):
        mock_get.return_value = _make_response(
            json_data={"memories": [{"memoryId": "mem-1"}, {"memoryId": "mem-2"}]}
        )
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["success"] is True
        assert result["totalMemories"] == 2
        assert result["memories"][0]["memoryId"] == "mem-1"
        url = mock_get.call_args[0][0]
        assert url.endswith("/v1/spaces/sp-1/memories")

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_list_response(self, mock_get):
        mock_get.return_value = _make_response(json_data=[{"memoryId": "mem-1"}])
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["memories"] == [{"memoryId": "mem-1"}]

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_empty(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))
        assert result["memories"] == []
        assert result["totalMemories"] == 0

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_with_params(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        tool._run(
            space_id="sp-1",
            status_filter="COMPLETED",
            sort_by="created_at",
            sort_order="DESCENDING",
        )
        params = mock_get.call_args[1]["params"]
        assert params["statusFilter"] == "COMPLETED"
        assert params["sortBy"] == "created_at"
        assert params["sortOrder"] == "DESCENDING"

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_include_content_true(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        tool._run(space_id="sp-1", include_content=True)
        params = mock_get.call_args[1]["params"]
        assert params["includeContent"] == "true"

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_include_content_false_omits_param(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        tool._run(space_id="sp-1", include_content=False)
        params = mock_get.call_args[1]["params"]
        assert "includeContent" not in params

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_no_params_by_default(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memories": []})
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        tool._run(space_id="sp-1")
        params = mock_get.call_args[1]["params"]
        assert params == {}

    @patch(f"{_MODULE}.requests.get")
    def test_list_memories_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_get.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemListMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="nonexistent"))
        assert result["success"] is False
        assert "Failed to list memories" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemRetrieveMemoriesTool
# ---------------------------------------------------------------------------

NDJSON_WITH_RESULTS = "\n".join(
    [
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-1", "boundary": "START"}}),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c-1",
                            "chunkText": "CrewAI is a framework",
                            "memoryId": "mem-1",
                        },
                        "relevanceScore": 0.95,
                        "memoryIndex": 0,
                    }
                }
            }
        ),
        json.dumps({"memoryDefinition": {"memoryId": "mem-1", "spaceId": "sp-1"}}),
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-1", "boundary": "END"}}),
    ]
)

NDJSON_EMPTY = json.dumps(
    {"resultSetBoundary": {"resultSetId": "rs-empty", "boundary": "START"}}
)

NDJSON_WITH_ABSTRACT = "\n".join(
    [
        json.dumps({"resultSetBoundary": {"resultSetId": "rs-2", "boundary": "START"}}),
        json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c-2",
                            "chunkText": "Some text",
                            "memoryId": "mem-2",
                        },
                        "relevanceScore": 0.8,
                        "memoryIndex": 0,
                    }
                }
            }
        ),
        json.dumps({"abstractReply": {"text": "This is a summary"}}),
    ]
)

SSE_FORMAT = "\n".join(
    [
        "event: message",
        'data: {"resultSetBoundary": {"resultSetId": "rs-sse"}}',
        "",
        "event: message",
        'data: {"retrievedItem": {"chunk": {"chunk": '
        '{"chunkId": "c-sse", "chunkText": "SSE text", '
        '"memoryId": "mem-sse"}, "relevanceScore": 0.9, '
        '"memoryIndex": 0}}}',
    ]
)


class TestRetrieveMemoriesTool:
    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_with_results(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="What is CrewAI?",
                space_ids=["sp-1"],
                wait_for_indexing=False,
            )
        )
        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkId"] == "c-1"
        assert result["results"][0]["chunkText"] == "CrewAI is a framework"
        assert result["results"][0]["relevanceScore"] == 0.95
        assert result["resultSetId"] == "rs-1"
        assert len(result["memories"]) == 1

    def test_retrieve_empty_space_ids(self):
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(query="test", space_ids=[]))
        assert result["success"] is False
        assert "At least one space ID" in result["error"]

    def test_retrieve_filters_empty_strings(self):
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(query="test", space_ids=["", ""], wait_for_indexing=False)
        )
        assert result["success"] is False

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_with_abstract_reply(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_ABSTRACT)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="test",
                space_ids=["sp-1"],
                wait_for_indexing=False,
            )
        )
        assert result["success"] is True
        assert "abstractReply" in result
        assert result["abstractReply"]["text"] == "This is a summary"

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_sse_format(self, mock_post):
        mock_post.return_value = _make_response(text=SSE_FORMAT)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="test",
                space_ids=["sp-1"],
                wait_for_indexing=False,
            )
        )
        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkId"] == "c-sse"

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_wait_for_indexing_timeout(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_EMPTY)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="test",
                space_ids=["sp-1"],
                wait_for_indexing=True,
                max_wait_seconds=0.1,
                poll_interval=0.05,
            )
        )
        assert result["success"] is True
        assert result["totalResults"] == 0
        assert "No results found" in result.get("message", "")

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_configurable_wait_params(self, mock_post):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_response(text=NDJSON_EMPTY)

        mock_post.side_effect = side_effect
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="test",
                space_ids=["sp-1"],
                wait_for_indexing=True,
                max_wait_seconds=0.15,
                poll_interval=0.05,
            )
        )
        assert result["totalResults"] == 0
        assert call_count >= 2

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_with_post_processor(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        tool._run(
            query="test",
            space_ids=["sp-1"],
            reranker_id="reranker-1",
            llm_id="llm-1",
            relevance_threshold=0.5,
            llm_temperature=0.3,
            chronological_resort=True,
            wait_for_indexing=False,
        )
        body = mock_post.call_args[1]["json"]
        assert "postProcessor" in body
        pp = body["postProcessor"]
        assert "ChatPostProcessorFactory" in pp["name"]
        cfg = pp["config"]
        assert cfg["reranker_id"] == "reranker-1"
        assert cfg["llm_id"] == "llm-1"
        assert cfg["relevance_threshold"] == 0.5
        assert cfg["llm_temp"] == 0.3
        assert cfg["chronological_resort"] is True

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_with_metadata_filter(self, mock_post):
        """metadata_filter is attached to every spaceKey server-side."""
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        filter_expr = "CAST(val('$.category') AS TEXT) = 'feat'"
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        tool._run(
            query="new features",
            space_ids=["sp-1", "sp-2"],
            metadata_filter=filter_expr,
            wait_for_indexing=False,
        )
        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"] == [
            {"spaceId": "sp-1", "filter": filter_expr},
            {"spaceId": "sp-2", "filter": filter_expr},
        ]

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_without_metadata_filter_omits_filter_key(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        tool._run(query="test", space_ids=["sp-1"], wait_for_indexing=False)
        body = mock_post.call_args[1]["json"]
        assert body["spaceKeys"] == [{"spaceId": "sp-1"}]

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_ndjson_accept_header(self, mock_post):
        mock_post.return_value = _make_response(text=NDJSON_WITH_RESULTS)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        tool._run(query="test", space_ids=["sp-1"], wait_for_indexing=False)
        headers = mock_post.call_args[1]["headers"]
        assert headers["Accept"] == "application/x-ndjson"

    @patch(f"{_MODULE}.requests.post")
    def test_retrieve_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        mock_post.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(
            tool._run(
                query="test",
                space_ids=["sp-1"],
                wait_for_indexing=False,
            )
        )
        assert result["success"] is False
        assert "Failed to retrieve memories" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemGetMemoryTool
# ---------------------------------------------------------------------------


class TestGetMemoryTool:
    @patch(f"{_MODULE}.requests.get")
    def test_get_memory_with_text_content(self, mock_get):
        meta_resp = _make_response(
            json_data={
                "memoryId": "mem-1",
                "processingStatus": "COMPLETED",
                "contentType": "text/plain",
            }
        )
        content_resp = _make_response(
            text="Hello world",
            headers={"Content-Type": "text/plain"},
        )
        mock_get.side_effect = [meta_resp, content_resp]
        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1", include_content=True))
        assert result["success"] is True
        assert result["memory"]["memoryId"] == "mem-1"
        assert result["content"] == "Hello world"
        assert result["contentEncoding"] == "text"
        assert mock_get.call_count == 2
        urls = [c[0][0] for c in mock_get.call_args_list]
        assert urls[0].endswith("/v1/memories/mem-1")
        assert urls[1].endswith("/v1/memories/mem-1/content")

    @patch(f"{_MODULE}.requests.get")
    def test_get_memory_binary_content_base64_encoded(self, mock_get):
        """Binary content (e.g. PDF) is base64-encoded so it JSON-serializes."""
        raw_bytes = b"%PDF-fake-content"
        meta_resp = _make_response(
            json_data={
                "memoryId": "mem-pdf",
                "processingStatus": "COMPLETED",
                "contentType": "application/pdf",
            }
        )
        content_resp = _make_response(
            content=raw_bytes,
            headers={"Content-Type": "application/pdf"},
        )
        mock_get.side_effect = [meta_resp, content_resp]
        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-pdf", include_content=True))
        assert result["success"] is True
        assert result["contentEncoding"] == "base64"
        assert base64.b64decode(result["content"]) == raw_bytes

    @patch(f"{_MODULE}.requests.get")
    def test_get_memory_without_content(self, mock_get):
        mock_get.return_value = _make_response(json_data={"memoryId": "mem-1"})
        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1", include_content=False))
        assert result["success"] is True
        assert "content" not in result
        assert mock_get.call_count == 1

    @patch(f"{_MODULE}.requests.get")
    def test_get_memory_content_fetch_error_sets_content_error(self, mock_get):
        """Failures on the /content endpoint must not fail the whole call."""
        meta_resp = _make_response(
            json_data={"memoryId": "mem-1", "processingStatus": "PROCESSING"}
        )
        content_resp = _make_response(
            raise_for_status=requests.RequestException("content not available")
        )
        mock_get.side_effect = [meta_resp, content_resp]
        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1", include_content=True))
        assert result["success"] is True
        assert "content" not in result
        assert "contentError" in result
        assert "Failed to fetch content" in result["contentError"]

    @patch(f"{_MODULE}.requests.get")
    def test_get_memory_metadata_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_get.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="nonexistent"))
        assert result["success"] is False
        assert "Failed to get memory" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemDeleteMemoryTool
# ---------------------------------------------------------------------------


class TestDeleteMemoryTool:
    @patch(f"{_MODULE}.requests.delete")
    def test_delete_success(self, mock_delete):
        mock_delete.return_value = _make_response(json_data={})
        tool = GoodMemDeleteMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1"))
        assert result["success"] is True
        assert result["memoryId"] == "mem-1"
        assert "deleted" in result["message"].lower()
        headers = mock_delete.call_args[1]["headers"]
        assert "Content-Type" not in headers

    @patch(f"{_MODULE}.requests.delete")
    def test_delete_not_found(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_delete.side_effect = requests.HTTPError(response=mock_resp)
        tool = GoodMemDeleteMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="nonexistent"))
        assert result["success"] is False
        assert "Failed to delete memory" in result["error"]


class TestPackageExports:
    EXPECTED_TOOLS = {
        "GoodMemCreateMemoryTool",
        "GoodMemCreateSpaceTool",
        "GoodMemDeleteMemoryTool",
        "GoodMemDeleteSpaceTool",
        "GoodMemGetMemoryTool",
        "GoodMemGetSpaceTool",
        "GoodMemListEmbeddersTool",
        "GoodMemListMemoriesTool",
        "GoodMemListSpacesTool",
        "GoodMemRetrieveMemoriesTool",
        "GoodMemUpdateSpaceTool",
    }

    def test_all_tools_exported_from_package(self):
        import crewai_goodmem

        for tool_name in self.EXPECTED_TOOLS:
            assert hasattr(crewai_goodmem, tool_name), (
                f"{tool_name} must be exported from crewai_goodmem"
            )

        assert tool_name in crewai_goodmem.__all__
        assert isinstance(crewai_goodmem.__version__, str)
        assert hasattr(crewai_goodmem, "wait_for_memories_completed")


class TestWaitForMemoriesCompleted:
    @patch(f"{_MODULE}.requests.get")
    def test_empty_list_returns_immediately(self, mock_get):
        from crewai_goodmem import wait_for_memories_completed

        result = wait_for_memories_completed(
            [], base_url=BASE_URL, api_key=API_KEY, verify_ssl=False
        )
        assert result == {}
        mock_get.assert_not_called()

    @patch(f"{_MODULE}.requests.get")
    def test_all_completed_first_round(self, mock_get):
        from crewai_goodmem import wait_for_memories_completed

        mock_get.return_value = _make_response(
            json_data={"memoryId": "mem-1", "processingStatus": "COMPLETED"}
        )
        result = wait_for_memories_completed(
            ["mem-1", "mem-2"],
            base_url=BASE_URL,
            api_key=API_KEY,
            verify_ssl=False,
            interval=0.01,
        )
        assert result == {"mem-1": "COMPLETED", "mem-2": "COMPLETED"}
        assert mock_get.call_count == 2

    @patch(f"{_MODULE}.time.sleep")
    @patch(f"{_MODULE}.requests.get")
    def test_polls_until_completed(self, mock_get, mock_sleep):
        from crewai_goodmem import wait_for_memories_completed

        responses = [
            _make_response(json_data={"processingStatus": "PENDING"}),
            _make_response(json_data={"processingStatus": "PROCESSING"}),
            _make_response(json_data={"processingStatus": "COMPLETED"}),
        ]
        mock_get.side_effect = responses
        result = wait_for_memories_completed(
            ["mem-1"],
            base_url=BASE_URL,
            api_key=API_KEY,
            verify_ssl=False,
            interval=0.01,
        )
        assert result == {"mem-1": "COMPLETED"}
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch(f"{_MODULE}.requests.get")
    def test_failed_status_is_terminal(self, mock_get):
        from crewai_goodmem import wait_for_memories_completed

        mock_get.return_value = _make_response(json_data={"processingStatus": "FAILED"})
        result = wait_for_memories_completed(
            ["mem-1"], base_url=BASE_URL, api_key=API_KEY, verify_ssl=False
        )
        assert result == {"mem-1": "FAILED"}

    @patch(f"{_MODULE}.requests.get")
    def test_timeout_raises(self, mock_get):
        from crewai_goodmem import wait_for_memories_completed

        mock_get.return_value = _make_response(
            json_data={"processingStatus": "PENDING"}
        )
        with pytest.raises(TimeoutError, match="Still pending"):
            wait_for_memories_completed(
                ["mem-1"],
                base_url=BASE_URL,
                api_key=API_KEY,
                verify_ssl=False,
                timeout=0.05,
                interval=0.01,
            )

    @patch(f"{_MODULE}.requests.get")
    def test_dedupes_input(self, mock_get):
        from crewai_goodmem import wait_for_memories_completed

        mock_get.return_value = _make_response(
            json_data={"processingStatus": "COMPLETED"}
        )
        result = wait_for_memories_completed(
            ["mem-1", "mem-1", "mem-2"],
            base_url=BASE_URL,
            api_key=API_KEY,
            verify_ssl=False,
        )
        assert result == {"mem-1": "COMPLETED", "mem-2": "COMPLETED"}
        assert mock_get.call_count == 2

    def test_missing_credentials_raises_value_error(self):
        from crewai_goodmem import wait_for_memories_completed

        with pytest.raises(ValueError, match="GOODMEM"):
            wait_for_memories_completed(["mem-1"])
