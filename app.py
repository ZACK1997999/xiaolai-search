import streamlit as st
import random

# 1. 这是一个极其简单的“数据库” (用列表模拟)
# 以后这里会换成真正的AI搜索，现在先用几句李笑来的话代替
try:
    with open("data.txt", "r", encoding="utf-8") as f:
        content = f.read()
        # 把一大段文本，按“换行符”切分成一句句，存进列表
        quotes = content.split("\n") 
        quotes = [q for q in quotes if q.strip()]
except FileNotFoundError:
    quotes = []
    st.error("出错啦！找不到 data.txt 文件。请确认它和 app.py 在同一个文件夹里！")
# 2. 搭建网页骨架 (拼图的边框)
st.title("我的第一个AI搜索神器")
st.write(f"📚目前收录 {len(quotes)} 条李笑来金句。")
# --- 新增功能区：侧边栏 (Sidebar) ---
# st.sidebar 是 Streamlit 的一个拼图块，专门用来放侧边菜单
st.sidebar.header("功能区")
# 功能 1: 随机抽取 (手气不错)
if st.sidebar.button("🎲 随机来一句"):
    if quotes:
        lucky_quote = random.choice(quotes) # 从列表里随机选一个
        st.success("✨ 命运给你的指引：")
        st.markdown(f"### {lucky_quote}")
    else:
        st.warning("数据库是空的哦！")
# 3. 输入 (Input)
query = st.text_input("请输入关键词 (例如：时间、学习、拼图):")

# 4. 按钮与逻辑 (Loop & Condition)
if st.button("开始搜索"):
    if query: # 如果用户输入了东西
        st.write(f"正在搜索：{query} ...")
        
        # --- 核心逻辑开始 ---
        found_results = []
        for sentence in quotes:
            # 如果关键词在句子里面
           if query.lower() in sentence.lower():
                found_results.append(sentence)
        # --- 核心逻辑结束 ---
        
        # 5. 输出 (Output)
        if found_results:
            st.success(f"找到了 {len(found_results)} 条结果：")
            for result in found_results:
                st.markdown(f"> {result}") # 显示结果
        else:
            st.warning("没有找到相关内容，换个词试试？")
    else:
        st.error("你还没输入关键词呢！")