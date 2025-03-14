import streamlit as st

from unsloth import FastLanguageModel
from transformers import AutoTokenizer


repo_name = "mrpojam/Llama3.2-1B-De2Fr-Translation"  

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = FastLanguageModel.from_pretrained(repo_name, device_map="cuda")[0]


data_prompt = """Translate the following German text to French:

### Input:
{}

### Output:
{}"""


def translate(request):
    prompt = tokenizer(
    [
        data_prompt.format(
            request,
            "",
        )
    ], return_tensors = "pt").to("cuda")
    
    result = model.generate(**prompt, max_new_tokens = 128, use_cache = True)
    decoded = tokenizer.batch_decode(result)
    result = decoded[0].split("Output:")[1].split("<|end_of_text|>")[0].replace("\n", "")
    return result

st.title("German2French Translation")

text = st.text_area("Enter your German text to translate", "")

if st.button("Translate"):
    if text:
        response = translate(text)
        try:
            translation = response
            st.subheader("Translation:")
            st.write(translation)
        except:
            st.error(f"Error: {response.text}")
    else:
        st.warning("Please enter text to translate.")
