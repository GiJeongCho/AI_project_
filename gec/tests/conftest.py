import pytest
from dotenv import load_dotenv


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--env-file",
        action="store",
        default=None,
        dest="env_file",
        help="Set env file path, default is None",
    )


def pytest_configure(config: pytest.Config):
    if env_file := config.getoption("env_file"):
        load_dotenv(str(env_file))
