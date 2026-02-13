from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    repo_intelligence: bool = True
    research_intelligence: bool = True
    mapping_engine: bool = True
    patch_generation: bool = True
    validation: bool = True

    benchmark_timeout: int = 300
    max_retries: int = 2

    log_level: str = "INFO"

    class Config:
        env_prefix = "SC_"


settings = Settings()
