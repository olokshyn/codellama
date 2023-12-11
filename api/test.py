from pydantic_settings import BaseSettings

from llama import Llama


class Settings(BaseSettings):
    ckpt_dir: str = "CodeLlama-7b-Instruct"
    tokenizer_path: str = "CodeLlama-7b-Instruct/tokenizer.model"
    temperature: float = 0.2
    top_p: float = 0.95
    max_seq_len: int = 2000
    max_batch_size: int = 8
    max_gen_len: int | None = None


settings = Settings()


model = Llama.build(
    ckpt_dir=settings.ckpt_dir,
    tokenizer_path=settings.tokenizer_path,
    max_seq_len=settings.max_seq_len,
    max_batch_size=settings.max_batch_size,
)

results = model.chat_completion(
    [
        [
            {
                "role": "system",
                "content": "Specify an arithmetic expression to evaluate.",
            },
            {"role": "user", "content": "7 + 3"},
        ]
    ],
    max_gen_len=settings.max_gen_len,
    temperature=settings.temperature,
    top_p=settings.top_p,
)
print(results)