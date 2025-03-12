from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from unsloth import FastLanguageModel

app = FastAPI()

repo_name = "mrpojam/Llama-De-Fr"  # Change this!

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = FastLanguageModel.from_pretrained(repo_name, device_map="cuda")[0]



class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    result = model.generate(**request.text, max_new_tokens = 32, use_cache = True)
    result = "test output"
    result = result.split("Output:")[1].split("<|end_of_text|>")[0].replace("\n", "")
    return {"translation": result}
