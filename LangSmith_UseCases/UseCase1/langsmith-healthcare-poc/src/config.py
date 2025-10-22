from dotenv import load_dotenv
import os


load_dotenv()


class Settings:
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "default")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()