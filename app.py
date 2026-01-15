import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import re

# ================= 配置区域 =================
# 🔴 请在这里填入你的真实 Key，你的朋友在网页上是看不到这个的
# 注意：不要把这个文件发给陌生人，否则他们能看到你的 Key
# ================= 配置区域 =================
try:
    # 尝试从 Streamlit 的云端“保险箱”获取 Key
    MY_HIDDEN_KEY = st.secrets["DEEPSEEK_API_KEY"]
except FileNotFoundError:
    st.error("未找到密钥！请配置 .streamlit/secrets.toml 或在云端设置 Secrets。")
    st.stop()
# ===========================================
# ===========================================

# --- 安全加载 YouTube 模块 ---
YOUTUBE_AVAILABLE = False
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    if not hasattr(YouTubeTranscriptApi, 'get_transcript'):
        raise ImportError
    YOUTUBE_AVAILABLE = True
except:
    YOUTUBE_AVAILABLE = False

# ================= 第一部分：Preply 风格精准测试 =================

def run_vocab_test():
    st.header("📈 英语词汇量评估")
    st.caption("为了让 AI 助教更懂你，请先完成两步快速测试。")
    st.info("请只勾选你**确实认识**（能说出中文意思）的单词。")
    
    step1_words = [
        "red", "bus", "salt", "rabbit", "hammer", 
        "sudden", "barely", "attend", "defend", "modest",
        "justice", "specialize", "harvest", "threshold", "mechanic",
        "ambiguous", "magnitude", "reinforce", "profound", "allegation",
        "manifestation", "conspiracy", "indigenous", "hypothesis", "pragmatic",
        "ubiquitous", "ephemeral", "meticulous", "exacerbate", "scrutinize",
        "esoteric", "vicarious", "obsequious", "idiosyncrasy", "sycophant"
    ]
    
    if 'test_stage' not in st.session_state:
        st.session_state['test_stage'] = 1
    
    # --- 阶段 1 ---
    if st.session_state['test_stage'] == 1:
        st.subheader("第一步：快速定位")
        cols = st.columns(5)
        selected_step1 = []
        for i, word in enumerate(step1_words):
            with cols[i % 5]:
                if st.checkbox(word, key=f"s1_{word}"):
                    selected_step1.append(word)
        
        st.write("---")
        if st.button("继续下一步", type="primary"):
            # 简单定级逻辑
            if len(selected_step1) < 10: st.session_state['temp_level'] = 'basic'
            elif len(selected_step1) < 20: st.session_state['temp_level'] = 'intermediate'
            else: st.session_state['temp_level'] = 'advanced'
            
            st.session_state['test_stage'] = 2
            st.rerun()

    # --- 阶段 2 ---
    elif st.session_state['test_stage'] == 2:
        st.subheader("第二步：精准校准")
        level = st.session_state.get('temp_level', 'intermediate')
        
        if level == 'basic':
            step2_words = ["cousin", "leather", "shelf", "pure", "shout", "dust", "belief", "pale", "wander", "squeeze", "curious", "bunch", "terror", "faint", "weed"]
            base_score = 1000; multiplier = 150
        elif level == 'intermediate':
            step2_words = ["subtle", "distinct", "prohibit", "adequate", "consult", "guarantee", "confront", "precious", "resign", "inherit", "scatter", "courage", "bloom", "polish", "frown"]
            base_score = 4000; multiplier = 300
        else:
            step2_words = ["cynical", "eloquent", "hinder", "plausible", "tedious", "rigorous", "subsequent", "integration", "proposition", "adverse", "mitigate", "consensus", "intriguing", "viability", "fluctuation"]
            base_score = 8000; multiplier = 500
            
        st.write(f"正在校准... 请勾选你认识的单词：")
        cols2 = st.columns(5)
        selected_step2 = []
        for i, word in enumerate(step2_words):
            with cols2[i % 5]:
                if st.checkbox(word, key=f"s2_{word}"):
                    selected_step2.append(word)
        
        st.write("---")
        if st.button("生成我的学习档案", type="primary"):
            vocab_size = base_score + (len(selected_step2) * multiplier)
            st.session_state['user_vocab_size'] = vocab_size
            
            # 生成 Prompt
            if vocab_size < 3000:
                desc = "初学者 (词汇量约3000)。请提取所有非基础的生词、常用短语。"
            elif vocab_size < 6000:
                desc = "中级学习者 (词汇量约5000，四级水平)。请略过简单的词，重点挖掘四六级难度的词、地道短语和熟词生义。"
            elif vocab_size < 10000:
                desc = "中高级学习者 (词汇量约8000，雅思/托福水平)。请只提取学术词汇、习语搭配、以及深层的熟词生义。"
            else:
                desc = "高阶学习者 (词汇量10000+)。请只挖掘极其罕见的生僻词、文学性词汇、以及隐喻用法。"
                
            st.session_state['user_profile_prompt'] = desc
            st.rerun()

# ================= 第二部分：AI 智能挖掘 =================

class SmartMiner:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def get_youtube_text(self, url):
        if not YOUTUBE_AVAILABLE: return "❌ YouTube 组件不可用"
        try:
            video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1].split("?")[0]
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return " ".join([t['text'] for t in transcript])
        except Exception as e:
            return f"Error: {e}"

    def analyze_text_with_ai(self, text, user_profile):
        system_prompt = f"""
        你是一个专业的英语私教。
        【用户画像】：{user_profile}
        
        你的任务是从文本中挖掘适合该用户的学习素材。请提取以下三类：
        1. **生词** (Words)。
        2. **短语** (Phrases)。
        3. **熟词生义** (Polysemy)。

        请严格以 JSON 格式返回列表，格式如下：
        [
            {{
                "word": "原型",
                "type": "生词" 或 "短语" 或 "熟词生义",
                "definition": "中文释义(必须对应文中的具体含义)",
                "context": "包含该词的完整原句"
            }}
        ]
        """
        user_prompt = f"【待分析文本】(前5000字符):\n{text[:5000]}"

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            if "```" in content:
                content = re.search(r'\[.*\]', content, re.DOTALL).group()
            
            data = json.loads(content)
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], list): return data[key]
            return data
        except Exception as e:
            st.error(f"AI 思考出错: {e}")
            return []

# ================= 主页面 UI =================

def main_tool_page():
    # 1. 侧边栏：这里展示你的联系方式
    st.sidebar.header("关于项目")
    st.sidebar.info("📢 内容为持续更新中\n\n💬 微信: **lifeaka7**")
    
    # 2. 检查是否完成测试
    if 'user_vocab_size' not in st.session_state:
        run_vocab_test()
        return

    st.set_page_config(page_title="AI 英语私教", layout="wide")
    st.title("🧠 AI 英语私教 (好友内测版)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"👤 用户档案: **{st.session_state['user_vocab_size']} 词**")
    with col2:
        if st.button("🔄 重测水平"):
            del st.session_state['user_vocab_size']
            st.session_state['test_stage'] = 1
            st.rerun()

    # 使用隐藏的 Key
    miner = SmartMiner(MY_HIDDEN_KEY)

    tab1, tab2 = st.tabs(["📄 文本挖掘", "📺 YouTube 挖掘"])
    
    raw_text = ""
    with tab1:
        txt = st.text_area("粘贴英文内容:", height=150, placeholder="粘贴文章...")
        if st.button("开始分析 (文本)"): raw_text = txt
            
    with tab2:
        if YOUTUBE_AVAILABLE:
            url = st.text_input("视频链接:")
            if st.button("开始分析 (视频)"): 
                raw = miner.get_youtube_text(url)
                if "Error" not in raw: raw_text = raw
                else: st.error(raw)
        else:
            st.warning("YouTube 组件不可用")

    if raw_text:
        if len(raw_text) < 10:
            st.warning("内容太短了")
        else:
            with st.spinner("🧠 AI 正在分析..."):
                results = miner.analyze_text_with_ai(raw_text, st.session_state['user_profile_prompt'])
            
            if results:
                st.balloons()
                st.write(f"### 🎯 挖掘结果 ({len(results)} 个)")
                df = pd.DataFrame(results)
                
                st.dataframe(
                    df, 
                    column_config={"word": "词汇", "type": "类型", "definition": "释义", "context": "原句"},
                    use_container_width=True
                )
                
                # Anki 格式
                anki_df = pd.DataFrame()
                anki_df['Front'] = df.apply(lambda x: f"<b>{x['word']}</b> <small style='color:grey'>[{x['type']}]</small><br><br>{x['context']}", axis=1)
                anki_df['Back'] = df['definition']
                
                csv = anki_df.to_csv(index=False, header=False).encode('utf-8')
                st.download_button("📥 下载 Anki 文件 (.csv)", csv, "anki_cards.csv", "text/csv")

if __name__ == "__main__":
    main_tool_page()