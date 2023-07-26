import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from datasets import load_dataset


# Check if GPU is available
if torch.cuda.is_available():
    device = 0
else:
    # If GPU is not available, use CPU
    device = torch.device("cpu")


# model_name = "facebook/bart-large-mnli"
# nlp_pipeline = pipeline("zero-shot-classification", model=model_name, device=device)
# st.title("USPTO Patentability Score")


@st.cache_data()
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
patent_info = df[["patent_number", "title"]]
patent_info["patent_info"] = patent_info.apply(
    lambda row: f"{row['patent_number']} - {row['title']}", axis=1
)

patent_number = patent_info["patent_info"].drop_duplicates().reset_index(drop=True)

st.sidebar.subheader("Select the Patent:")
make_choice = st.sidebar.selectbox("", patent_number)

with st.form("patent-form"):
    submitted = st.form_submit_button(label="Submit")

if submitted:
    selected_patent_number = make_choice.split(" - ")[0]
    st.subheader(f"Patent Application Number: {selected_patent_number}")

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    decision = df["decision"].loc[df["patent_number"] == selected_patent_number]
    X_train = decision.to_string(index=False)
    results = classifier(X_train, truncation=True)

    for result in results:
        score = result["score"]
        st.write("The Patentability Score is:", score)

    abstract = df["abstract"].loc[df["patent_number"] == selected_patent_number]
    st.subheader("Abstract:")
    st.info(abstract.iloc[0])

    claims = df["claims"].loc[df["patent_number"] == selected_patent_number]
    st.subheader("Claim:")
    st.info(claims.iloc[0])
