import streamlit as st
import requests

PRESET_TONES = [
    "干货科普风", "闺蜜种草风", "职场进阶风", "低调实战风",
    "数码评测风", "极客硬核风", "生活好物风", "商业变现风",
    "自我提升风", "企业服务风", "移动办公风", "运营干货风",
    "网感娱乐风", "独立开发风", "养生关怀风", "销售销冠风",
    "跨境出海风", "沉浸学习风", "求职面试风", "高效工具风",
    "✏️ 自定义风格..."
]

st.title("📕 企业级闭环营销 Agent (PoC)")

topic = st.text_input("💡 你想写什么主题？")
tone_select = st.selectbox("🎭 请选择文案风格", PRESET_TONES)

tone = tone_select
if tone_select == "✏️ 自定义风格...":
    custom_tone = st.text_input("✏️ 请输入你想要的风格描述", placeholder="例如：小红书玄学风、知乎盐选风...")
    tone = custom_tone if custom_tone else "干货科普风"

if st.button("🚀 开始生成文案"):
    if not topic.strip():
        st.warning("请输入主题")
    else:
        with st.spinner("Agent 工作流已启动 (正在进行 检索->生成->审核 闭环)..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/generate_copy",
                    json={"topic": topic, "tone": tone}
                )
                if response.status_code == 200:
                    result = response.json()
                    data = result.get('data', {})

                    content = data.get('content', '文案生成完毕。')
                    iterations = data.get('iterations_run', '多')

                    st.success(f"✨ 生成成功！(Agent 内部进行了 {iterations} 轮自我反思与重写)")
                    st.markdown("---")
                    st.markdown(content)
                else:
                    st.error(f"服务端错误详情: {response.json().get('detail', response.text)}")

            except Exception as e:
                st.error(f"网络请求失败：{e}")