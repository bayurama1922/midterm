import streamlit as st
from transformers import BertTokenizer, EncoderDecoderModel

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
model = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")

def generate_summary(article):
    input_ids = tokenizer.encode(article, return_tensors='pt')
    summary_ids = model.generate(input_ids,
                min_length=20,
                max_length=80, 
                num_beams=10,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95)

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# Streamlit UI
st.title("Text Summarization App")

article_input = st.text_area("Input your article here:")

if st.button("Summarize"):
    if article_input.strip():
        summary = generate_summary(article_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please input an article to summarize.")