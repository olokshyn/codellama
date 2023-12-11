import logging

from fastapi import FastAPI, Body
from pydantic_settings import BaseSettings
from uvicorn import run

from llama import Llama


class Settings(BaseSettings):
    ckpt_dir: str = "CodeLlama-7b-Instruct"
    tokenizer_path: str = "CodeLlama-7b-Instruct/tokenizer.model"
    temperature: float = 0.2
    top_p: float = 0.95
    max_seq_len: int = 2000
    max_batch_size: int = 8
    max_gen_len: int | None = None


logging.basicConfig(level=logging.INFO)


settings = Settings()
app = FastAPI()


generator = Llama.build(
    ckpt_dir=settings.ckpt_dir,
    tokenizer_path=settings.tokenizer_path,
    max_seq_len=settings.max_seq_len,
    max_batch_size=settings.max_batch_size,
)


@app.post("/")
def root(instructions: list[list[dict[str, str]]] = Body(...)):
    logging.info(f"Received instructions: {instructions}")
    results = generator.chat_completion(
        instructions,
        max_gen_len=settings.max_gen_len,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )
    logging.info(f"Generated results: {results}")
    return results


if __name__ == "__main__":
    run(app, host="localhost", port=8000)
