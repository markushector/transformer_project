from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    version: str
    model_version: str


settings = Settings()

