import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tools import sourceformat as sf
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
nltk.download('stopwords')


# ===config===
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

with st.expander("Before you start", expanded = True):
    tab1, tab2, tab3, tab4 = st.tabs(["Prologue", "Steps", "Requirements", "Download Visualization"])
    with tab1:
            st.write("")
        
    with tab2:
        st.text("1. Put your file. Choose your preferred column to analyze (if CSV).")
        st.text("2. Choose your preferred method to count the words and decide how many top words you want to include or remove.")
        st.text("3. Finally, you can visualize your data.")
        st.error("This app includes lemmatization and stopwords. Currently, we only offer English words.", icon="💬")
        
        with tab3:
            st.code("""
            +----------------+------------------------+
            |     Source     |       File Type        |           
            +----------------+------------------------+
            | Scopus         | Comma-separated values |      
            |                | (.csv)                 |     
            +----------------+------------------------|   
            | Lens.org       | Comma-separated values |                                  
            |                | (.csv)                 |                                  
            +----------------+------------------------|                                 
            | Other          | .csv/ .txt(full text)  |                                  
            +----------------+------------------------|                                  
            | Hathitrust     | .json                  |                                  
            +----------------+------------------------+
            """, language=None)
        
    with tab4:
        st.subheader(':blue[WordCloud Download]', anchor=False)
        st.write("Right-click image and click \"Save-as\"")

st.header("Wordcloud", anchor=False)
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
@st.cache_data(ttl=3600)
def read_txt(intext):
    return (intext.read()).decode()

@st.cache_data(ttl=3600)
def conv_txt(extype):
    if("PMID" in (uploaded_file.read()).decode()):
        uploaded_file.seek(0)
        papers = sf.medline(uploaded_file)
        print(papers)
        return papers
    col_dict = {'TI': 'Title',
            'SO': 'Source title',
            'DE': 'Author Keywords',
            'DT': 'Document Type',
            'AB': 'Abstract',
            'TC': 'Cited by',
            'PY': 'Year',
            'ID': 'Keywords Plus',
            'rights_date_used': 'Year'}
    uploaded_file.seek(0)
    papers = pd.read_csv(uploaded_file, sep='\t')
    
    #if text just has one column (or is not csv) return nothing
    if(len(papers.columns)==1):
        return

    if("htid" in papers.columns):
        papers = sf.htrc(papers)
    papers.rename(columns=col_dict, inplace=True)
    print(papers)
    return papers

#===csv reading===
@st.cache_data(ttl=3600)
def upload(file):
    papers = pd.read_csv(uploaded_file)
    if "About the data" in papers.columns[0]:
        papers = sf.dim(papers)
        col_dict = {'MeSH terms': 'Keywords',
        'PubYear': 'Year',
        'Times cited': 'Cited by',
        'Publication Type': 'Document Type'
        }
        papers.rename(columns=col_dict, inplace=True)
    
    return papers

#===Read data===
uploaded_file = st.file_uploader('', type=['txt','csv'], on_change=reset_all)
    
if uploaded_file is not None:
    
    tab1, tab2, tab3 = st.tabs(["📈 Generate visualization", "📃 Reference", "⬇️ Download Help"])
    
    with tab1:
        c1, c2 = st.columns(2)

        
        with c1:
            max_font = st.number_input("Max Font Size", min_value = 1, value = 100)
            max_words = st.number_input("Max Word Count", min_value = 1, value = 250)
            background = st.selectbox("Background color", ["white","black"])

        
        with c2:
            words_to_remove = st.text_input("Remove specific words. Separate words by semicolons (;)")
            filterwords = words_to_remove.split(';')
            image_width = st.number_input("Image width", value = 400)
            image_height = st.number_input("Image height", value = 200)
            scale = st.number_input("Scale", value = 1)
        
        try:
            extype = get_ext(uploaded_file)

            if extype.endswith(".txt"):
                
                try:
                    texts = conv_txt(uploaded_file)
                    colcho = c1.selectbox("Choose Column", list(texts))
                    fulltext = " ".join(list(texts[colcho]))
                    fulltext = remove_stopwords(fulltext)
                    

                except:
                    fulltext = read_txt(uploaded_file)
                
                if st.button("Submit"):
                    
                    
                    wordcloud = WordCloud(max_font_size = max_font,
                    max_words = max_words,
                    background_color=background,
                    stopwords = filterwords,
                    height = image_height,
                    width = image_width,
                    scale = scale).generate(fulltext)
                    img = wordcloud.to_image()

                    with st.container(border=True):
                        st.image(img, use_container_width=True)

            elif extype.endswith(".csv"):
                texts = upload(uploaded_file)

                colcho = c1.selectbox("Choose Column", list(texts))

                fullcolumn = " ".join(list(texts[colcho]))
                
                fullcolumn = remove_stopwords(fullcolumn)

                if st.button("Submit"):

                    wordcloud = WordCloud(max_font_size = max_font,
                    max_words = max_words,
                    background_color=background,
                    stopwords = filterwords,
                    height = image_height,
                    width = image_width,
                    scale = scale).generate(fullcolumn)
                    img = wordcloud.to_image()

                    with st.container(border=True):
                        st.image(img, use_container_width=True)

        except Exception as e:
            st.write(e)