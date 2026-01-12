import os

# 注意：如果你刚才加了那个 hf-mirror 的代码，请删掉，换成下面这两句
# 把 7890 改成你实际的端口号
#os.environ['http_proxy'] = 'http://127.0.0.1:7890'
#os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 下面才是 import streamlit ...
import streamlit as st
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import torch

# --- 1. 页面基本设置 ---
app_name = "我的财富自由外挂 🚀"  # 把名字存进一个叫 app_name 的盒子里
st.title(app_name)              # 告诉网页：去把那个盒子里的字显示出来

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
        # 1. 读取整个文件内容（不要在这里 split）
        with open("data.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # 2. 【关键修改】使用 LangChain 进行智能切分
        # chunk_size=500: 每个片段大约500字，保证内容完整
        # chunk_overlap=50: 前后重叠50字，防止把一句话切断
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        
        # 这里的 sentences 现在变成了“长段落列表”，不再是短句子了
        sentences = text_splitter.split_text(full_text)
        
        if not sentences:
            return [], None

        # 3. 变成向量 (这一步不用变)
        embeddings = model.encode(sentences, convert_to_tensor=True)
        
        return sentences, embeddings
        
    except FileNotFoundError:
        return [], None

sentences, sentence_embeddings = load_and_encode_data()

if not sentences:
    st.error("出错啦！找不到 data.txt，或者文件里没内容。")
    st.stop()

st.success(f"已加载 {len(sentences)} 条李笑来的智慧。")
# === 新增：AI 大脑函数 ===
def get_ai_answer(user_query, context_list):
    """
    user_query: 用户的提问
    context_list: 搜出来的几段李笑来的原文
    """
    # 1. 配置 DeepSeek 的钥匙（记得把下面的 sk-xxx 换成你刚才申请的）
    client = OpenAI(
        api_key=st.secrets["DEEPSEEK_API_KEY"], 
        base_url="https://api.deepseek.com"  # DeepSeek 的官方地址
    )

    # 2. 把几段原文拼起来，变成一大段背景资料
    context_str = "\n\n".join(context_list)

    # 3. 构造提示词 (Prompt) - 这一步决定了 AI 的说话风格
    system_prompt = """
   你就是李笑来。
    请基于下方的【参考资料】回答用户的【问题】。
    
    你的语言风格要求：
    1. 强调“长期主义”、“践行”、“时间的朋友”、“注意力”等概念。
    2. 语气要理性、冷静，甚至有点“硬核”，不要只会说好听的鸡汤。
    3. 经常使用这样的句式：“所谓的……本质上……”、“这一点非常重要”。
    4. 如果资料里没有答案，就直接说不知道，不要编造，要诚实。
    
    请用Markdown格式输出，重点部分加粗。
    """

    user_message = f"""
    【参考资料】：
    {context_str}

    【用户问题】：
    {user_query}
    """

    # 4. 发送给 DeepSeek
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用 DeepSeek V3 模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 思考时出错了：{e}"

# --- 4. 搜索界面 ---
query = st.text_input("🔍 请输入你的问题 (比如：如何实现财富自由？):")

if st.button("AI 搜索"):
    if query:
        # 1. 算出问题和文档的相似度 (这部分保持不变)
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(3, len(sentences)))

        # === 修改重点开始 ===
        
        # 准备一个空列表，用来装搜到的好内容
        found_contexts = []
        
        st.write("---")
        st.subheader("🔍 搜索结果与 AI 解读")

        # 循环提取搜到的内容
        for score, idx in zip(top_results.values, top_results.indices):
            if score > 0.25:  # 只要相似度大于 0.25 的
                content = sentences[idx]
                found_contexts.append(content) # 把内容收集起来
                
                #把原文折叠起来，想看的人可以点开看
                with st.expander(f"参考原文 (相似度 {score:.2f})"):
                    st.text(content)

        # 关键时刻：如果有搜到内容，就发给 AI
        if found_contexts:
            with st.spinner("AI 正在阅读原文并为你总结..."):
                final_answer = get_ai_answer(query, found_contexts)
                st.success(final_answer) # 绿框显示 AI 的回答
        else:
            st.warning("在他的文章里没找到相关内容，换个关键词试试？")