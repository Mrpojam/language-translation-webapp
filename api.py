from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from unsloth import FastLanguageModel

app = FastAPI()

repo_name = "mrpojam/Llama3.2-1B-De2Fr-Translation"  

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = FastLanguageModel.from_pretrained(repo_name, device_map="cuda")[0]



class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    data_prompt = """Translate the following German text to French:

    ### Input:
    {}

    ### Output:
    {}"""

    prompt = tokenizer(
    [
        data_prompt.format(
            request,
            "",
        )
    ], return_tensors = "pt").to("cuda")
    result = model.generate(**prompt, max_new_tokens = 128, use_cache = True)
    decoded = tokenizer.batch_decode(result)
    translation = decoded.split("Output:")[1].split("<|end_of_text|>")[0].replace("\n", "")
    return {"translation": translation}
