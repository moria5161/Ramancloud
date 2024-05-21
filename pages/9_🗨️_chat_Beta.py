import time
import streamlit as st

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# '''
# 星火认知大模型v3.5的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# 星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
# 星火认知大模型v3.5的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
# '''

def stream_data(input_text):
    for word in input_text:
        yield word
        time.sleep(0.01)


SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_APP_ID = '4cbb7e2a'
SPARKAI_API_SECRET = 'ZGE3NjE2Y2Q3OGYxZjc1ODNiNzMwNWMx'
SPARKAI_API_KEY = 'a5fe84683a2ac2256cb040f51cb55e8a'
SPARKAI_DOMAIN = 'generalv3.5'

try:
    from dotenv import load_dotenv
except ImportError:
    raise RuntimeError(
        'Python environment for SPARK AI is not completely set up: required package "python-dotenv" is missing.') from None

load_dotenv()

spark = ChatSparkLLM(
    spark_api_url=SPARKAI_URL,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN,
    streaming=True,
)
handler = ChunkPrintHandler()

messages_list = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.success('''**Ask anything about Raman spectral analysis!**  
           Such as: i am analyzing a Raman spectrum of an unknown solution. There is a peak at nearly 1000cm^-1, what is this?''')

for msg in st.session_state.messages[:-1]:
    st.chat_message(name=msg.role).write(msg.content)

message_container = st.container()


if prompt := st.chat_input("Ask something about Raman spectroscopy"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    message_container.chat_message("user").write(prompt)

    response = spark.generate([st.session_state.messages], callbacks=[handler])
    st.session_state.messages.append(ChatMessage(role="user", content=response.generations[0][0].text))

    message_container.chat_message("assistant").write_stream(stream_data(response.generations[0][0].text))


