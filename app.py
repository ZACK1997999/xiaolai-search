import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. 页面基本设置 ---
st.title("李笑来 AI 语义搜索 🧠")
st.write("输入你的困惑，让 AI 帮你从李笑来的文章里找答案。")

# --- 2. 加载 AI 模型 (这步最慢，所以要缓存起来) ---
@st.cache_resource
def load_model():
    # 这里我们选一个支持中文的多语言模型
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

with st.spinner('正在启动 AI 大脑，第一次运行需要下载模型，请稍等...'):
    model = load_model()

# --- 3. 读取并处理数据 ---
@st.cache_data
def load_and_encode_data():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        # 去掉空行和太短的句子
        sentences = [line.strip() for line in lines if len(line.strip()) > 5]
        
        if not sentences:
            return [], None
            
        # 关键步骤：把所有句子变成向量 (Embedding)
        # 这一步会让电脑把文字理解成数字
        embeddings = model.encode(sentences, convert_to_tensor=True)
        return sentences, embeddings
    except FileNotFoundError:
        return [], None

sentences, sentence_embeddings = load_and_encode_data()

if not sentences:
    st.error("出错啦！找不到 data.txt，或者文件里没内容。")
    st.stop()

st.success(f"已加载 {len(sentences)} 条李笑来的智慧。")

# --- 4. 搜索界面 ---
query = st.text_input("🔍 请输入你的问题 (比如：如何实现财富自由？):")

if st.button("AI 搜索"):
    if query:
        # 1. 把用户的问题也变成向量
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 2. 计算相似度 (Cosine Similarity) - 这就是 AI 的魔法
        # 也就是算一下你的问题和数据库里的每一句话有多像
        cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        
        # 3. 找出分数最高的 5 个结果
        top_results = torch.topk(cos_scores, k=min(5, len(sentences)))
        
        st.write("---")
        st.subheader("AI 认为最相关的答案：")
        
        for score, idx in zip(top_results.values, top_results.indices):
            # score 是相似度分数 (0到1之间，越大越像)
            if score > 0.3: # 只要分数大于 0.3 的结果
                st.markdown(f"**相似度 {score:.2f}**")
                st.info(sentences[idx])
            else:
                # 如果分数太低，说明没找到很好的
                pass
                
    else:
        st.warning("请输入问题！")