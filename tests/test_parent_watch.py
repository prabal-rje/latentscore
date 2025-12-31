import os

from app.parent_watch import PARENT_FD_ENV, PARENT_PID_ENV, parse_parent_watch_env


def test_parse_parent_watch_env_reads_fd_and_pid() -> None:
    read_fd, write_fd = os.pipe()
    try:
        env = {
            PARENT_FD_ENV: str(read_fd),
            PARENT_PID_ENV: str(os.getpid()),
        }
        config = parse_parent_watch_env(env)
        assert config.fd == read_fd
        assert config.pid == os.getpid()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_parse_parent_watch_env_ignores_invalid_values() -> None:
    env = {
        PARENT_FD_ENV: "not-a-fd",
        PARENT_PID_ENV: "-1",
    }
    config = parse_parent_watch_env(env)
    assert config.fd is None
    assert config.pid is None
