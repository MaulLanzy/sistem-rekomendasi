import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="UBM Course Recommender AI",
    page_icon="🎓",
    layout="wide"
)

# --- 1. SETUP GOOGLE GEMINI AI ---
# Fungsi untuk memanggil otak AI
def ask_gemini(query):
    try:
        # Coba ambil API Key dari Secrets
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Persona AI: Asisten Kampus
            prompt = f"""
            Kamu adalah Asisten Akademik Virtual untuk Universitas Bunda Mulia (UBM). 
            Gaya bicaramu santai, ramah, mendukung mahasiswa, dan kekinian (tapi tetap sopan).
            
            User bertanya: "{query}"
            
            Jawablah pertanyaan tersebut. Jika itu tentang saran akademik, berikan motivasi. 
            Jika pertanyaan umum, jawab dengan ringkas dan jelas.
            """
            response = model.generate_content(prompt)
            return response.text
        else:
            return "⚠️ API Key belum dipasang. Silakan atur di Streamlit Secrets."
    except Exception as e:
        return f"Maaf, AI sedang istirahat. Error: {str(e)}"

# --- 2. DATA & LOGIKA ---

KEYWORD_MAPPING = {
    "menggambar": "desain visual art seni fotografi kreatif sketsa ilustrasi grafis",
    "jualan": "marketing bisnis manajemen pemasaran retail sales perdagangan kewirausahaan",
    "ngoding": "teknologi informasi sistem komputer data algoritma programming python web",
    "hitung": "akuntansi statistika matematika ekonomi keuangan pajak finance analisis",
    "jalan-jalan": "pariwisata hospitality hotel tour travel guide tourism wisata",
    "masak": "food beverage tata boga kitchen pastry culinary",
    "game": "game development interactive design programming unity multimedia",
    "tidur": "santai istirahat kesehatan mental psikologi",
    "duit": "investasi keuangan bisnis entrepreneur kaya",
}

PROGRAM_DESCRIPTIONS = {
    "Informatika": "Mempelajari pengembangan software, teknologi jaringan, dan komputasi cerdas.",
    "Sistem Informasi": "Menggabungkan ilmu komputer dengan manajemen bisnis untuk sistem perusahaan.",
    "Manajemen": "Fokus pada pengelolaan bisnis, strategi pemasaran, dan kepemimpinan.",
    "Akuntansi": "Ahli dalam pencatatan, analisis, dan pelaporan keuangan bisnis.",
    "Ilmu Komunikasi": "Strategi penyampaian pesan efektif melalui media digital dan humas.",
    "Hospitality dan Pariwisata": "Menyiapkan profesional perhotelan, kuliner, dan manajemen wisata.",
    "Desain Komunikasi Visual": "Solusi komunikasi visual yang kreatif, artistik, dan inovatif.",
    "Bahasa Inggris": "Komunikasi profesional global melalui bahasa dan budaya.",
    "Bahasa Mandarin": "Bahasa dan budaya Tiongkok untuk bisnis internasional.",
    "Bisnis Digital": "Teknologi digital dalam strategi bisnis modern.",
    "Data Science": "Mengolah Big Data menjadi wawasan untuk prediksi.",
    "Psikologi": "Mempelajari perilaku manusia dan proses mental."
}

@st.cache_data
def load_data():
    try:
        # Ganti nama file sesuai data kamu
        df = pd.read_csv("List Mata Kuliah UBM.xlsx - Sheet1.csv")
        df = df.dropna(subset=['Course'])
        df['combined_features'] = df['Course'] + " " + df['Program']
        return df
    except FileNotFoundError:
        st.error("File CSV tidak ditemukan. Pastikan sudah diupload!")
        return pd.DataFrame()

def get_program_description(program_name):
    for key, desc in PROGRAM_DESCRIPTIONS.items():
        if key in program_name:
            return desc
    return "Jurusan unggulan pencetak profesional handal."

def get_course_advice(course_name):
    course_lower = course_name.lower()
    if any(x in course_lower for x in ['matematika', 'statistik', 'akuntansi', 'keuangan']):
        return {"tip": "💡 **Tips:** Pahami konsep dasar, jangan cuma hafal rumus. Latihan soal kuncinya!"}
    elif any(x in course_lower for x in ['coding', 'algoritma', 'data', 'web']):
        return {"tip": "💻 **Tips:** Praktek langsung (ngoding) lebih efektif daripada baca teori. Jangan takut error!"}
    elif any(x in course_lower for x in ['desain', 'gambar', 'art', 'sketsa']):
        return {"tip": "🎨 **Tips:** Perbanyak lihat referensi (Pinterest/Behance) dan bangun portofolio."}
    elif any(x in course_lower for x in ['bisnis', 'manajemen', 'marketing']):
        return {"tip": "📊 **Tips:** Pelajari studi kasus nyata perusahaan dan latih skill presentasi."}
    else:
        return {"tip": "📝 **Tips:** Catat poin penting dosen dan aktif bertanya di kelas."}

# --- 3. UI & CSS ---
def local_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, p, label { color: #ffffff !important; }
    
    /* Kartu Hasil */
    .result-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #667eea;
    }
    .result-card h3 { color: #31333F !important; margin: 0; }
    .result-card p { color: #31333F !important; margin: 5px 0 0 0; }
    
    /* Chatbot Box */
    .chat-box {
        background-color: #1E1E1E;
        border: 1px solid #4CAF50;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LOGIKA UTAMA ---
def main_app():
    local_css()
    
    # Inisialisasi Bookmark
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = []

    df = load_data()
    if df.empty: return

    # Sidebar
    with st.sidebar:
        st.header("🔍 Filter Pencarian")
        jurusan_list = ["Semua"] + sorted(df['Program'].unique().tolist())
        selected_jurusan = st.selectbox("Pilih Jurusan:", jurusan_list)
        
        semester_list = ["Semua"] + sorted(df['Semester'].unique().tolist())
        selected_semester = st.selectbox("Pilih Semester:", semester_list)
        
        st.divider()
        st.subheader("🔖 Bookmark Saya")
        if st.session_state.bookmarks:
            for item in st.session_state.bookmarks:
                st.write(f"- {item['Course']} (Sem {item['Semester']})")
            if st.button("Hapus Semua Bookmark"):
                st.session_state.bookmarks = []
                st.rerun()
        else:
            st.info("Belum ada mata kuliah disimpan.")

    # Header Utama
    st.title("🎓 UBM Course Recommender AI")
    st.markdown("Cari mata kuliah berdasarkan minatmu, atau **tanya apa saja pada AI!**")

    # Input User
    user_query = st.text_input("Apa yang ingin kamu pelajari?", placeholder="Contoh: Saya suka menggambar, atau Cara belajar efektif?")
    search_clicked = st.button("Cari / Tanya AI")

    if search_clicked and user_query:
        # 1. Preprocessing (Negation & Mapping)
        query_processed = user_query.lower()
        
        # Hapus kata negatif ("tidak suka hitung" -> "tidak suka")
        negatives = ["tidak suka", "benci", "gak suka", "anti"]
        for neg in negatives:
            if neg in query_processed:
                parts = query_processed.split(neg)
                if len(parts) > 1:
                    unwanted_word = parts[1].strip().split()[0]
                    query_processed = query_processed.replace(unwanted_word, "")
        
        # Keyword Expansion
        expanded_query = query_processed
        for key, value in KEYWORD_MAPPING.items():
            if key in query_processed:
                expanded_query += " " + value

        # 2. Filter Data Frame
        filtered_df = df.copy()
        if selected_jurusan != "Semua":
            filtered_df = filtered_df[filtered_df['Program'] == selected_jurusan]
        if selected_semester != "Semua":
            filtered_df = filtered_df[filtered_df['Semester'] == selected_semester]

        # 3. Algoritma TF-IDF
        if not filtered_df.empty:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(filtered_df['combined_features'])
            query_vec = vectorizer.transform([expanded_query])
            cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            filtered_df['Similarity Score'] = cosine_scores
            # Ambil Top 5 yang skornya > 15%
            recs = filtered_df[filtered_df['Similarity Score'] > 0.15].sort_values(by='Similarity Score', ascending=False).head(5)
            
            # --- LOGIKA HYBRID (DATABASE vs AI) ---
            
            # Jika ditemukan Mata Kuliah yang relevan
            if not recs.empty:
                st.success(f"Ditemukan {len(recs)} Mata Kuliah yang cocok!")
                
                for idx, row in recs.iterrows():
                    prog_desc = get_program_description(row['Program'])
                    advice = get_course_advice(row['Course'])
                    
                    # Kartu Hasil
                    st.markdown(f"""
                    <div class="result-card">
                        <h3>{row['Course']}</h3>
                        <p>🎓 {row['Program']} | 📅 Sem {row['Semester']} | ⭐ {int(row['Similarity Score']*100)}% Match</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"💡 Info & Tips: {row['Course']}"):
                        st.write(f"**Jurusan:** {prog_desc}")
                        st.info(advice['tip'])
                        
                        # Tombol Save
                        if st.button(f"🔖 Simpan {row['Course']}", key=f"save_{idx}"):
                            st.session_state.bookmarks.append(row.to_dict())
                            st.rerun()

            else:
                # JIKA TIDAK ADA MATKUL -> LEMPAR KE GEMINI AI
                st.warning("Hmm, tidak ada mata kuliah spesifik yang cocok di database...")
                with st.spinner("Sedang bertanya ke Asisten AI..."):
                    ai_response = ask_gemini(user_query)
                
                st.markdown(f"""
                <div class="chat-box">
                    <h3>🤖 Jawaban Asisten AI:</h3>
                    <p>{ai_response}</p>
                </div>
                """, unsafe_allow_html=True)

        # 4. FITUR CHATBOT EKSTRA
        # Jika user bertanya pertanyaan umum (bukan cari matkul), kita tampilkan juga jawaban AI di bawah
        # Cek apakah query berbentuk pertanyaan
        if any(q in user_query.lower() for q in ["apa", "bagaimana", "kenapa", "tips", "cara"]):
             if filtered_df.empty or recs.empty:
                 pass # Sudah dijawab di atas
             else:
                 with st.spinner("AI sedang menambahkan tips tambahan..."):
                    ai_response = ask_gemini(user_query)
                    st.markdown("---")
                    st.subheader("🤖 Pendapat Asisten AI:")
                    st.write(ai_response)

if __name__ == "__main__":
    main_app()
