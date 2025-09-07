import streamlit as st
import streamlit.components.v1 as components
import nltk
import spacy
import pytextrank
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer, T5ForConditionalGeneration, T5Tokenizer
nltk.download('punkt')

#===config===
st.set_page_config(
    page_title="Coconut",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_style = """
            <style>
            #MainMenu 
            {visibility: hidden;}
            footer {visibility: hidden;}
            [data-testid="collapsedControl"] {display: none}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.popover("🔗 Menu"):
    st.page_link("https://www.coconut-libtool.com/", label="Home", icon="🏠")
    st.page_link("pages/1 Scattertext.py", label="Scattertext", icon="1️⃣")
    st.page_link("pages/2 Topic Modeling.py", label="Topic Modeling", icon="2️⃣")
    st.page_link("pages/3 Bidirected Network.py", label="Bidirected Network", icon="3️⃣")
    st.page_link("pages/4 Sunburst.py", label="Sunburst", icon="4️⃣")
    st.page_link("pages/5 Burst Detection.py", label="Burst Detection", icon="5️⃣")
    st.page_link("pages/6 Keywords Stem.py", label="Keywords Stem", icon="6️⃣")
    st.page_link("pages/7 Sentiment Analysis.py", label="Sentiment Analysis", icon="7️⃣")
    st.page_link("pages/8 Shifterator.py", label="Shifterator", icon="8️⃣")
    st.page_link("pages/9 Summarization.py", label = "Summarization",icon ="9️⃣")
    st.page_link("pages/10 WordCloud.py", label = "WordCloud", icon = "🔟")
    
st.header("Summarization", anchor=False)
st.subheader('Put your file here...', anchor=False)

#========unique id========
@st.cache_resource(ttl=3600)
def create_list():
    l = [1, 2, 3]
    return l

l = create_list()
first_list_value = l[0]
l[0] = first_list_value + 1
uID = str(l[0])

@st.cache_data(ttl=3600)
def get_ext(uploaded_file):
    extype = uID+uploaded_file.name
    return extype

#===clear cache===
def reset_all():
    st.cache_data.clear()

#===text reading===
def read_txt(intext):
    return (intext.read()).decode()

#===csv reading===
def read_csv(uploaded_file):
    fulltexts = pd.read_csv(uploaded_file)
    fulltexts.rename(columns={fulltexts.columns[0]: "texts"}, inplace = True)
    return fulltexts
    

#===Read data===
uploaded_file = st.file_uploader('', type=['txt','csv'], on_change=reset_all)


if uploaded_file is not None:
    try:
        extype = get_ext(uploaded_file)

        if extype.endswith(".txt"):
            fulltext = read_txt(uploaded_file)
        elif extype.endswith(".csv"):
            texts = read_csv(uploaded_file)

        #Menu
        
        method = st.selectbox("Method",("Extractive","Abstractive"))
        if method == "Abstractive":
            ab_method = st.selectbox("Abstractive method", ("Pegasus x-sum","FalconsAI t5","facebook/bart-large-cnn"))
            min_length = st.number_input("Minimum length", min_value = 0)
            max_length = st.number_input("Maximum length", min_value = 1)        

        if method == "Extractive":
            ex_method = st.selectbox("Extractive method", ("t5","PyTextRank"))
            if ex_method == "PyTextRank":
                phrase_limit = st.number_input("Phrase limit", min_value = 0)
                sentence_limit = st.number_input("Sentence limit", min_value = 0)
            elif ex_method == "t5" or ex_method == "FalconsAI t5":
                min_length = st.number_input("Minimum length", min_value = 0)
                max_length = st.number_input("Maximum length", min_value = 1)                  

        

        if st.button("Submit", on_click=reset_all):
        
            tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "⬇️ Download Help"])
                
            with tab1:
                
                def SpacyRank(text):
                    nlp = spacy.load("en_core_web_sm")
                    nlp.add_pipe("textrank")
                    doc = nlp(text)
                    summary = ""
                    for sent in doc._.textrank.summary(limit_phrases = phrase_limit, limit_sentences = sentence_limit):
                        summary+=str(sent) + '\n'
                    return summary

                def t5summ(text):
                    model = T5ForConditionalGeneration.from_pretrained('t5-small')
                    tokenizer = T5Tokenizer.from_pretrained('t5-small')
                    
                    input_text = "summarize: " + text
                    input_ids = tokenizer.encode(input_text,return_tensors='pt')
                    
                    summed = model.generate(input_ids, max_length = max_length, min_length = min_length)

                    summary = tokenizer.decode(summed[0],skip_special_tokens=True)     
                    return summary       

                def xsum(text):
                    model_name = "google/pegasus-xsum"
                    
                    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

                    summarizer = pipeline("summarization", 
                    model=model_name, 
                    tokenizer=pegasus_tokenizer, 
                    framework="pt")
                    
                    summed = summarizer(text, min_length = min_length, max_length = max_length)
                    summary = summed[0]["summary_text"]   

                    return summary                     

                def falcsum(text):
                    summarizer = pipeline("summarization",model = "Falconsai/text_summarization")
                    summed = summarizer(text, max_length = max_length, min_length = min_length, do_sample = False)                    
                    summary = summed[0]["summary_text"]
                    return summary

                #used for any other huggingface model not used above

                def transformersum(text,model):
                    summarizer = pipeline("summarization", model = model)
                    summed = summarizer(text, max_length = max_length, min_length = min_length, do_sample = False)
                    summary = summed[0]["summary_text"]
                    return summary
                

                def bulkScore(combined):
                    
                    scorelist = []

                    for column in range(len(combined)):
                        ref = combined[column][0]
                        cand = combined[column][1]
                    
                        BLEuscore = nltk.translate.bleu_score.sentence_bleu([ref], cand)
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                        rougescores = scorer.score(ref, cand)

                        Bscore = f"{BLEuscore:.2f}"
                        Rscore = f"{rougescores['rouge1'].fmeasure:.2f}"

                        scoreTuplet = Bscore, Rscore

                        scorelist.append(scoreTuplet)

                    return scorelist


                with st.spinner('Performing computations. Please wait ...'):
                
                    c1, c2 = st.columns([0.5,0.5], border=True)
                
                    if(extype.endswith(".txt")):

                        with c1:
                            if(extype.endswith(".txt")):
                                st.header("Original text")
                                with st.container(border=True):
                                    st.write(fulltext)

                            if method == "Extractive":
                                if(ex_method == "PyTextRank"):
                                    summary = SpacyRank(fulltext)
                                elif(ex_method == "t5"):
                                    summary = t5summ(fulltext)

                            elif method == "Abstractive":
                                if ab_method == "Pegasus x-sum":
                                    summary = xsum(fulltext)

                                elif ab_method == "FalconsAI t5":
                                    summary = t5summ(fulltext)
                                elif ab_method == "facebook/bart-large-cnn":
                                    summary = transformersum(fulltext,ab_method)
                        with c2:
                            
                            st.header("Summarized")
                            with st.container(border = True):
                                st.write(summary)
                            st.header("Performance scores")
                            with st.container(border = True):
                                
                                #performance metrics
                                reference = fulltext
                                candidate = summary      

                                BLEuscore = nltk.translate.bleu_score.sentence_bleu([reference], candidate)

                                scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                                rougescores = scorer.score(reference, candidate)

                                st.write(f"BLEU Score: {BLEuscore:.2f}")
                                st.write(f"ROUGE-1 F1 Score: {rougescores['rouge1'].fmeasure:.2f}")

                                text_file = summary
                                st.download_button(
                                    label = "Download Results",
                                    data=text_file,
                                    file_name="Summary.txt",
                                    mime="text\csv",
                                    on_click="ignore",)

                    elif(extype.endswith(".csv")):
                        if method == "Extractive":
                            if(ex_method == "PyTextRank"):
                                summaries = texts['texts'].apply(SpacyRank)
                                fullnsums = summaries.to_frame()
                                fullnsums['full'] = texts['texts']
                                fullnsums['combined'] = fullnsums.values.tolist()


                            elif(ex_method == "t5"):
                                summaries = texts['texts'].apply(t5summ)
                                fullnsums = summaries.to_frame()
                                fullnsums['full'] = texts['texts']
                                fullnsums['combined'] = fullnsums.values.tolist()
                                

                        elif method == "Abstractive":
                            if ab_method == "Pegasus x-sum":
                                summaries = texts['texts'].apply(xsum)
                                fullnsums = summaries.to_frame()
                                fullnsums['full'] = texts['texts']
                                fullnsums['combined'] = fullnsums.values.tolist()

                            elif ab_method == "FalconsAI t5":
                                summaries = texts['texts'].apply(falcsum)
                                fullnsums = summaries.to_frame()
                                fullnsums['full'] = texts['texts']
                                fullnsums['combined'] = fullnsums.values.tolist()

                        with c1:
                            st.header("Download bulk summarization results")

                            result = summaries.to_csv()
                            st.download_button(
                                label = "Download Results",
                                data = result,
                                file_name = "Summaries.csv",
                                mime="text\csv",
                                on_click = "ignore"
                            )

                        with c2:
                            st.header("Scores and summaries results")
                            scores  = pd.DataFrame.from_records(bulkScore(fullnsums.combined.to_list()),columns = ["BLEU","Rouge"])
                        
                            summariesscores = fullnsums.join(scores)

                            summariesscores.drop("combined", axis = 1, inplace = True)
                            summariesscores.rename(columns = {"texts":"summarized"}, inplace = True)

                            result2 = summariesscores.to_csv()

                            st.download_button(
                                label = "Download scores and results",
                                data = result2,
                                file_name = "ScoredSummaries.csv",
                                mime = "text\csv",
                                on_click = "ignore"
                            )

            #do this
            with tab2:
                st.write("")

            with tab3:
                st.header("Summarization result")
                st.write("Click the download button (example) to get the text file result")
                st.button(label = "Download Results")


    except Exception:
        st.error("Please ensure that your file is correct. Please contact us if you find that this is an error.", icon="🚨")
        st.stop()
