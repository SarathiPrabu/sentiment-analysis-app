import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

st.title("USPTO Patentability Score")

@st.cache(allow_output_mutation=True)
def load_data():
    dataset_dict = load_dataset(
        "HUPD/hupd",
        name="sample",
        data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
        icpr_label=None,
        train_filing_start_date="2016-01-01",
        train_filing_end_date="2016-01-31",
        val_filing_start_date="2016-02-22",
        val_filing_end_date="2016-02-29",
    )
    df = pd.DataFrame.from_dict(dataset_dict["train"])
    df = df[["patent_number", "title", "decision", "abstract", "claims", "filing_date"]]
    return df

df = load_data()

selected_patent_info = st.sidebar.selectbox("Select the Patent:", df["patent_number"])

with st.form("patent-form"):
    st.subheader("Patent Application Number:")
    selected_patent_number = st.selectbox("", df["patent_number"])

    submitted = st.form_submit_button(label="Submit")

if submitted:
    st.subheader(f"Patent Application Number: {selected_patent_number}")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    decision_text = df["decision"].loc[df["patent_number"] == selected_patent_number].to_string(index=False)
    results = classifier(decision_text, truncation=True)

    for result in results:
        score = result["score"]
        st.write("The Patentability Score is:", score)

    abstract = df["abstract"].loc[df["patent_number"] == selected_patent_number]
    st.subheader("Abstract:")
    st.info(abstract.iloc[0])

    claims = df["claims"].loc[df["patent_number"] == selected_patent_number]
    st.subheader("Claim:")
    st.info(claims.iloc[0])
