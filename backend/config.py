from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Market data
    polygon_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./finance_aggregator.db"

    # Redis (optional)
    redis_url: str = "redis://localhost:6379/0"

    # App
    app_env: str = "development"
    debug: bool = True
    cors_origins: str = "http://localhost:8000,http://localhost:3000"

    # Feature flags
    use_simulated_data: bool = True
    enable_ocr_worker: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        return v

    def cors_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_live(self) -> bool:
        return bool(self.polygon_api_key or (self.alpaca_api_key and self.alpaca_secret_key))


settings = Settings()
