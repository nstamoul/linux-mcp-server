from pydantic import SecretStr

from linux_mcp_server.connection.ssh import discover_ssh_password


def test_discover_ssh_password_not_configured(mocker):
    """Test that discover_ssh_password returns None when nothing is configured."""
    mocker.patch("linux_mcp_server.connection.ssh.CONFIG.ssh_password", SecretStr(""))

    result = discover_ssh_password()

    assert result is None


def test_discover_ssh_password_configured(mocker):
    """Test that discover_ssh_password returns the configured password."""
    mocker.patch("linux_mcp_server.connection.ssh.CONFIG.ssh_password", SecretStr("hunter2"))

    result = discover_ssh_password()

    assert result == "hunter2"
