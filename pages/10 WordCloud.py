import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
def read_txt(intext):
    return (intext.read()).decode()

#===csv reading===
def read_csv(uploaded_file):
    fulltexts = pd.read_csv(uploaded_file)
    fulltexts.rename(columns={fulltexts.columns[0]: "texts"}, inplace = True)
    return fulltexts
    

#===Read data===
uploaded_file = st.file_uploader('', type=['txt'], on_change=reset_all)


if uploaded_file is not None:
    try:
        extype = get_ext(uploaded_file)

        if extype.endswith(".txt"):
            fulltext = read_txt(uploaded_file)

            wordcloud = WordCloud().generate(fulltext)
            img = wordcloud.to_image()

            with st.container(border=True):
                st.image(img)

        elif extype.endswith(".csv"):
            texts = read_csv(uploaded_file)


        


    except Exception as e:
        st.write(e)