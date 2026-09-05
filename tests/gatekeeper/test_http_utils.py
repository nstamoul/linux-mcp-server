import pytest

from linux_mcp_server.gatekeeper.http_utils import GatekeeperHTTPError
from linux_mcp_server.gatekeeper.http_utils import parse_sse_events
from linux_mcp_server.gatekeeper.http_utils import post_json
from linux_mcp_server.gatekeeper.http_utils import post_maybe_sse


async def test_post_json_success(mocker):
    response = mocker.MagicMock()
    response.is_success = True
    response.json.return_value = {"ok": True}
    mock_client = mocker.AsyncMock()
    mock_client.post.return_value = response
    mocker.patch("linux_mcp_server.gatekeeper.http_utils.HTTP_CLIENT", mock_client)

    result = await post_json(
        provider="openai",
        url="https://example.com/v1/responses",
        headers={"Authorization": "Bearer test"},
        body={"model": "gpt-5.4"},
    )

    assert result == {"ok": True}
    mock_client.post.assert_awaited_once()


async def test_post_json_error(mocker):
    response = mocker.MagicMock()
    response.is_success = False
    response.status_code = 503
    response.text = "service unavailable"
    mock_client = mocker.AsyncMock()
    mock_client.post.return_value = response
    mocker.patch("linux_mcp_server.gatekeeper.http_utils.HTTP_CLIENT", mock_client)

    with pytest.raises(GatekeeperHTTPError, match="openai API error \\(503\\)"):
        await post_json(
            provider="openai",
            url="https://example.com/v1/responses",
            headers={},
            body={},
        )


class TestParseSSEEvents:
    def test_parses_data_lines_into_json(self):
        text = (
            "event: response.created\n"
            'data: {"type": "response.created"}\n'
            "\n"
            "event: response.completed\n"
            'data: {"type": "response.completed"}\n'
            "\n"
            "data: [DONE]\n"
        )

        events = parse_sse_events(text)

        assert events == [{"type": "response.created"}, {"type": "response.completed"}]

    def test_skips_non_json_data_lines(self):
        text = 'event: codex.keepalive\ndata: not-json\n\ndata: {"type": "ok"}\n'

        events = parse_sse_events(text)

        assert events == [{"type": "ok"}]

    def test_empty_body_returns_no_events(self):
        assert parse_sse_events("") == []


class TestPostMaybeSSE:
    async def test_json_content_type_returns_json_body_and_no_events(self, mocker):
        response = mocker.MagicMock()
        response.is_success = True
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {"ok": True}
        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = response
        mocker.patch("linux_mcp_server.gatekeeper.http_utils.HTTP_CLIENT", mock_client)

        json_body, events = await post_maybe_sse(
            provider="openai",
            url="https://example.com/v1/responses",
            headers={},
            body={},
        )

        assert json_body == {"ok": True}
        assert events is None

    async def test_event_stream_content_type_returns_events_and_no_json(self, mocker):
        response = mocker.MagicMock()
        response.is_success = True
        response.headers = {"content-type": "text/event-stream; charset=utf-8"}
        response.text = 'data: {"type": "response.completed"}\n\ndata: [DONE]\n'
        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = response
        mocker.patch("linux_mcp_server.gatekeeper.http_utils.HTTP_CLIENT", mock_client)

        json_body, events = await post_maybe_sse(
            provider="openai",
            url="https://example.com/v1/responses",
            headers={},
            body={},
        )

        assert json_body is None
        assert events == [{"type": "response.completed"}]

    async def test_error_response_raises_regardless_of_content_type(self, mocker):
        response = mocker.MagicMock()
        response.is_success = False
        response.status_code = 401
        response.text = "unauthorized"
        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = response
        mocker.patch("linux_mcp_server.gatekeeper.http_utils.HTTP_CLIENT", mock_client)

        with pytest.raises(GatekeeperHTTPError, match="openai API error \\(401\\)"):
            await post_maybe_sse(
                provider="openai",
                url="https://example.com/v1/responses",
                headers={},
                body={},
            )
