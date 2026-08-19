import argparse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from utils.gac_gen_call import *
from utils.gac_gen_utils import *


@asynccontextmanager
async def lifespan(app: FastAPI):

    config_api_server, norm_type_api_server, threshold_api_server = load_yaml_config(
        args.config_path
    )

    global model_actors_list, tokenizers, vocab_union, mapping_matrices, index_to_vocab, special_prefix_tokens_dict, byte_mappings_list, min_max_position_embeddings, model_name_list, primary_index, threshold

    (
        model_actors_list,
        tokenizers,
        vocab_union,
        mapping_matrices,
        index_to_vocab,
        special_prefix_tokens_dict,
        byte_mappings_list,
        min_max_position_embeddings,
        model_name_list,
        primary_index,
        threshold,
    ) = setup_model_actors_and_data(
        config_api_server, norm_type_api_server, threshold_api_server
    )

    yield


parser = argparse.ArgumentParser(
    description="A script that uses a config file for GaC ensemble."
)
parser.add_argument(
    "--config-path",
    type=str,
    default="configs/new_7b_4_model.yaml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The host address to bind to. Default is 0.0.0.0",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind to. Default is 8000"
)
args = parser.parse_args()

app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    messages_list: List[
        List[Dict]
    ]  # List of messages(conversations) for batch processing
    max_length: Optional[int] = Field(default=None)  # Optional maximum length
    max_new_tokens: Optional[int] = Field(default=50)  # Specifying maximum new tokens
    apply_chat_template: Optional[bool] = Field(default=False)
    # For early stopping
    until: Optional[List[str]] = Field(default=None)


@app.get("/status")
async def get_status():
    return {"status": "ready"}


@app.post("/api/generate/")
async def api_generate(request: GenerateRequest):
    chat_list = request.messages_list
    max_length = request.max_length
    max_new_tokens = request.max_new_tokens
    apply_chat_template = request.apply_chat_template
    until = request.until

    length_param = (
        {"max_length": max_length}
        if max_length is not None
        else {"max_new_tokens": max_new_tokens}
    )

    prepare_inputs = [
        model_actor.prepare_inputs_for_model.remote(
            chat_list, min_max_position_embeddings, apply_chat_template
        )
        for model_actor in model_actors_list
    ]
    models_inputs = ray.get(prepare_inputs)
    input_ids_0 = models_inputs[0]

    output = generate_ensemnble_response(
        model_actors_list=model_actors_list,
        model_name_list=model_name_list,
        tokenizers=tokenizers,
        vocab_union=vocab_union,
        mapping_matrices=mapping_matrices,
        index_to_vocab=index_to_vocab,
        special_prefix_tokens_dict=special_prefix_tokens_dict,
        byte_mappings_list=byte_mappings_list,
        primary_index=primary_index,
        threshold=threshold,
        until=until,
        **length_param,
    )

    generated_texts = extract_generated_texts(tokenizers[0], input_ids_0, output)

    logger.info(f"Generated text:{generated_texts}")

    return {"response": generated_texts}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
