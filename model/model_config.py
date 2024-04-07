from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):

    n_embd: int = 384
    batch_size: int = 64
    block_size: int = 256
    n_head: int = 6
    n_layer: int = 6
    #lr = 3e-4
    #dropout = 0.2
    #epochs = 1000