import os
import re
import fitz  # PyMuPDF
import google.generativeai as genai
import streamlit as st
from prompt import PROMPT_WORKAW
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

# --- โหลด Config ---
dotenv.load_dotenv()

# แก้ไขส่วนการโหลด API Key ให้รองรับทั้ง Local และ Streamlit Cloud
if 'GOOGLE_API_KEY' in st.secrets:
    GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
else:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# --- Config (Temperature 0 = แม่นยำที่สุด) ---
generation_config = {
    "temperature": 0.0,
    "top_p": 1.0, 
    "top_k": 32,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# --- 🔥 CSS ธีมโดเรม่อนสีฟ้า 🔥 ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #00A0E9 0%, #FFFFFF 100%);
}
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background-color: rgba(235, 245, 255, 0.8);
}
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: 10px;
    right: 10px;
    width: 200px;
    height: 200px;
    background-image: url('https://upload.wikimedia.org/wikipedia/en/b/bd/Doraemon_character.png');
    background-size: contain;
    background-repeat: no-repeat;
    z-index: 99;
    opacity: 0.95;
    pointer-events: none;
}
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.85) !important;
    border: 2px solid #00A1E9 !important;
    border-radius: 25px !important;
    box-shadow: 3px 3px 12px rgba(0,0,0,0.1);
}
h1 {
    color: #E60012 !important; 
    text-shadow: 2px 2px white;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- ระบบอ่านไฟล์แบบ Hybrid ---
@st.cache_resource
def load_pdf_data_hybrid(file_path):
    text_content = ""
    page_images_map = {} 
    
    # ตรวจสอบไฟล์ใน Folder ปัจจุบัน
    if os.path.exists(file_path):
        try:
            doc = fitz.open(file_path)
            for i, page in enumerate(doc):
                page_num = i + 1
                text = page.get_text()
                text_content += f"\n[--- Page {page_num} START ---]\n{text}\n[--- Page {page_num} END ---]\n"
                
                image_blocks = [b for b in page.get_text("blocks") if b[6] == 1]
                saved_images = []
                
                if image_blocks:
                    for img_block in image_blocks:
                        rect = fitz.Rect(img_block[:4])
                        if rect.width > 50 and rect.height > 50: 
                            pix_crop = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
                            saved_images.append(pix_crop.tobytes("png"))
                
                if not saved_images:
                    pix_full = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    saved_images.append(pix_full.tobytes("png"))

                if saved_images:
                    page_images_map[page_num] = saved_images
            return text_content, page_images_map
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return "", {}
    else:
        st.error(f"ไม่พบไฟล์ {file_path} กรุณาตรวจสอบชื่อไฟล์บน GitHub ว่าเป็น 'Graphic.pdf' (ตัว G ใหญ่) หรือไม่")
        return "", {}

# --- เรียกใช้งานข้อมูล PDF ---
# ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับใน GitHub เป๊ะๆ
pdf_filename = "Graphic.pdf"
pdf_text, pdf_hybrid_images = load_pdf_data_hybrid(pdf_filename)

# --- System Prompt ---
FULL_SYSTEM_PROMPT = f"""
{PROMPT_WORKAW}
(กฎเหล็ก: ตอบตาม Context เท่านั้น และใส่ [PAGE: เลขหน้า] ทุกครั้ง)
----------------------------------------
CONTEXT:
{pdf_text}
----------------------------------------
"""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=FULL_SYSTEM_PROMPT
)

# --- UI Streamlit ---
def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "ฮัลโหล! ผมคือโดเรม่อนบอท มีอะไรให้ช่วยเรื่องกราฟิกไหมครับ? ✨"}
    ]
    st.rerun()

st.title("🤖 Graphic Doraemon Bot ✨")

with st.sidebar:
    if st.button("🗑️ ล้างประวัติการคุย"):
        clear_history()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "content": "ฮัลโหล! ผมคือโดเรม่อนบอท มีอะไรให้ช่วยเรื่องกราฟิกไหมครับ? ✨"}
    ]

for msg in st.session_state["messages"]:
    avatar_icon = "👨‍💻" if msg["role"] == "user" else "🔵"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.write(msg["content"])
        if "image_list" in msg:
             for img_data in msg["image_list"]:
                st.image(img_data, use_container_width=True)

if prompt := st.chat_input():
    if not GOOGLE_API_KEY:
        st.error("กรุณาใส่ GOOGLE_API_KEY ใน Settings > Secrets ของ Streamlit ก่อนครับ")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="👨‍💻").write(prompt)

        try:
            history_api = []
            for msg in st.session_state["messages"]:
                if "content" in msg and msg["content"]:
                    role = "user" if msg["role"] == "user" else "model"
                    history_api.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            # ลบข้อความล่าสุดออกชั่วคราวเพื่อส่งผ่าน start_chat
            user_input = history_api.pop()["parts"][0]["text"]
            
            chat_session = model.start_chat(history=history_api)
            strict_prompt = f"{user_input}\n(ระบุเลขหน้า [PAGE: x] ให้ตรงกับ Tag ใน Context)"
            response = chat_session.send_message(strict_prompt)
            
            response_text = response.text
            page_match = re.search(r"\[PAGE:\s*(\d+)\]", response_text)
            images_to_show = []
            p_num = None

            if page_match:
                p_num = int(page_match.group(1))
                if p_num in pdf_hybrid_images:
                    images_to_show = pdf_hybrid_images[p_num]

            with st.chat_message("model", avatar="🔵"):
                st.write(response_text)
                for img in images_to_show:
                    st.image(img, caption=f"🖼️ รายละเอียดจากหน้า {p_num}", use_container_width=True)
            
            st.session_state["messages"].append({
                "role": "model", 
                "content": response_text, 
                "image_list": images_to_show
            })

        except Exception as e:
            st.error(f"ระบบขัดข้อง: {e}")
