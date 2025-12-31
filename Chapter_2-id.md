# Langkah Pertama dengan LangChain

Di bab sebelumnya, kita menjelajahi LLM dan memperkenalkan LangChain sebagai kerangka kerja yang kuat untuk membangun aplikasi bertenaga LLM. Kita membahas bagaimana LLM telah merevolusi pemrosesan bahasa alami dengan kemampuan mereka memahami konteks, menghasilkan teks seperti manusia, dan melakukan penalaran kompleks. Meskipun kemampuan ini mengesankan, kita juga memeriksa keterbatasannya—halusinasi, batasan konteks, dan kurangnya pengetahuan terkini.

Di bab ini, kita akan beralih dari teori ke praktik dengan membangun aplikasi LangChain pertama kita. Kita akan mulai dengan dasar-dasar: menyiapkan lingkungan pengembangan yang tepat, memahami komponen inti LangChain, dan membuat rantai sederhana. Dari sana, kita akan menjelajahi kemampuan yang lebih canggih, termasuk menjalankan model lokal untuk privasi dan efisiensi biaya serta membangun aplikasi multimodal yang menggabungkan teks dengan pemahaman visual. Di akhir bab ini, Anda akan memiliki fondasi yang kokoh dalam blok bangunan LangChain dan siap untuk membuat aplikasi AI yang semakin canggih di bab-bab selanjutnya.

Singkatnya, bab ini akan membahas topik-topik berikut:

- Menyiapkan dependensi
- Menjelajahi blok bangunan LangChain (antarmuka model, prompt dan template, serta LCEL)
- Menjalankan model lokal
- Aplikasi AI multimodal

:::note
Mengingat evolusi cepat baik LangChain maupun bidang AI secara lebih luas, kami menjaga contoh kode dan sumber daya terkini di repositori GitHub kami: [https://github.com/benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain).

Untuk pertanyaan atau bantuan pemecahan masalah, silakan buat isu di GitHub atau bergabunglah dengan komunitas Discord kami: [https://packt.link/lang](https://packt.link/lang).
:::

## Menyiapkan dependensi untuk buku ini

Buku ini menyediakan beberapa opsi untuk menjalankan contoh kode, mulai dari notebook cloud tanpa setup hingga lingkungan pengembangan lokal. Pilih pendekatan yang paling sesuai dengan tingkat pengalaman dan preferensi Anda. Bahkan jika Anda terbiasa dengan manajemen dependensi, harap baca petunjuk ini karena semua kode dalam buku ini akan bergantung pada instalasi lingkungan yang benar seperti yang diuraikan di sini.

Untuk memulai dengan cepat tanpa setup lokal, kami menyediakan notebook online siap pakai untuk setiap bab:

- **Google Colab**: Jalankan contoh dengan akses GPU gratis
- **Kaggle Notebooks**: Bereksperimen dengan dataset terintegrasi
- **Gradient Notebooks**: Akses opsi komputasi berkinerja lebih tinggi

Semua contoh kode yang Anda temukan dalam buku ini tersedia sebagai notebook online di GitHub di [https://github.com/benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain).

Notebook-notebook ini tidak memiliki semua dependensi yang dikonfigurasi sebelumnya tetapi, biasanya, beberapa perintah instalasi membuat Anda siap berjalan. Alat-alat ini memungkinkan Anda mulai bereksperimen segera tanpa khawatir tentang setup. Jika Anda lebih suka bekerja secara lokal, kami merekomendasikan menggunakan conda untuk manajemen lingkungan:

1. Instal Miniconda jika Anda belum memilikinya.
2. Unduh dari [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
3. Buat lingkungan baru dengan Python 3.11:
   ```
   conda create -n langchain-book python=3.11
   ```
4. Aktifkan lingkungan:
   ```
   conda activate langchain-book
   ```
5. Instal Jupyter dan dependensi inti:
   ```
   conda install jupyter
   pip install langchain langchain-openai jupyter
   ```
6. Luncurkan Jupyter Notebook:
   ```
   jupyter notebook
   ```

Pendekatan ini memberikan lingkungan yang bersih dan terisolasi untuk bekerja dengan LangChain. Untuk pengembang berpengalaman dengan alur kerja yang sudah mapan, kami juga mendukung:

- **pip dengan venv**: Instruksi di repositori GitHub
- **Kontainer Docker**: Dockerfile disediakan di repositori GitHub
- **Poetry**: File konfigurasi tersedia di repositori GitHub

Pilih metode yang paling nyaman bagi Anda, tetapi ingat bahwa semua contoh mengasumsikan lingkungan Python 3.10+ dengan dependensi yang tercantum dalam requirements.txt.

Untuk pengembang, Docker, yang menyediakan isolasi melalui kontainer, adalah opsi yang baik. Kelemahannya adalah menggunakan banyak ruang disk dan lebih kompleks daripada opsi lainnya. Untuk ilmuwan data, saya merekomendasikan Conda atau Poetry.

Conda menangani dependensi rumit dengan efisien, meskipun bisa sangat lambat di lingkungan besar. Poetry menyelesaikan dependensi dengan baik dan mengelola lingkungan; namun, tidak menangkap dependensi sistem.

Semua alat memungkinkan berbagi dan replikasi dependensi dari file konfigurasi. Anda dapat menemukan serangkaian instruksi dan file konfigurasi yang sesuai di repositori buku di [https://github.com/benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain).

Setelah selesai, pastikan Anda telah menginstal LangChain versi 0.3.17. Anda dapat memeriksanya dengan perintah `pip show langchain`.

:::note
Dengan laju inovasi yang cepat di bidang LLM, pembaruan pustaka sering terjadi. Kode dalam buku ini diuji dengan LangChain 0.3.17, tetapi versi yang lebih baru mungkin memperkenalkan perubahan. Jika Anda mengalami masalah saat menjalankan contoh:

- Buat isu di repositori GitHub kami
- Bergabunglah dalam diskusi di Discord di [https://packt.link/lang](https://packt.link/lang)
- Periksa errata di halaman Packt buku ini

Dukungan komunitas ini memastikan Anda akan dapat mengimplementasikan semua proyek terlepas dari pembaruan pustaka.
:::

## Penyiapan kunci API

Pendekatan LangChain yang agnostik penyedia mendukung berbagai penyedia LLM, masing‑masing dengan kekuatan dan karakteristik unik. Kecuali Anda menggunakan LLM lokal, untuk menggunakan layanan ini, Anda perlu mendapatkan kredensial autentikasi yang sesuai.

| Penyedia        | Variabel Lingkungan               | URL Penyiapan                                                            | Tingkat Gratis?   |
| --------------- | --------------------------------- | ------------------------------------------------------------------------ | ----------------- |
| OpenAI          | `OPENAI_API_KEY`                  | [platform.openai.com](https://platform.openai.com)                       | Tidak             |
| HuggingFace     | `HUGGINGFACEHUB_API_TOKEN`        | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Ya                |
| Anthropic       | `ANTHROPIC_API_KEY`               | [console.anthropic.com](https://console.anthropic.com)                   | Tidak             |
| Google AI       | `GOOGLE_API_KEY`                  | [ai.google.dev/gemini-api](https://ai.google.dev/gemini-api)             | Ya                |
| Google VertexAI | `Application Default Credentials` | [cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)         | Ya (dengan batas) |
| Replicate       | `REPLICATE_API_TOKEN`             | [replicate.com](https://replicate.com)                                   | Tidak             |

Tabel 2.1: Tabel referensi kunci API (ikhtisar)

Kebanyakan penyedia memerlukan kunci API, sementara penyedia cloud seperti AWS dan Google Cloud juga mendukung metode autentikasi alternatif seperti **Application Default Credentials** (**ADC**). Banyak penyedia menawarkan tingkat gratis tanpa memerlukan detail kartu kredit, membuatnya mudah untuk memulai.

:::note
Lihat _Lampiran_ di akhir buku untuk mempelajari cara mendapatkan kunci API untuk OpenAI, Hugging Face, Google, dan penyedia lainnya.
:::

Untuk mengatur kunci API di lingkungan, dalam Python, kita dapat mengeksekusi baris berikut:

```python
import os
os.environ["OPENAI_API_KEY"] = "<token Anda>"
```

Di sini, `OPENAI_API_KEY` adalah kunci lingkungan yang sesuai untuk OpenAI. Mengatur kunci di lingkungan Anda memiliki keuntungan tidak perlu memasukkannya sebagai parameter dalam kode Anda setiap kali menggunakan model atau integrasi layanan.

Anda juga dapat mengekspos variabel‑variabel ini di lingkungan sistem Anda dari terminal. Di Linux dan macOS, Anda dapat mengatur variabel lingkungan sistem dari terminal menggunakan perintah `export`:

```
export OPENAI_API_KEY=<token Anda>
```

Untuk mengatur variabel lingkungan secara permanen di Linux atau macOS, Anda perlu menambahkan baris sebelumnya ke file `~/.bashrc` atau `~/.bash_profile`, lalu memuat ulang shell menggunakan perintah `source ~/.bashrc` atau `source ~/.bash_profile`.

Untuk pengguna Windows, Anda dapat mengatur variabel lingkungan dengan mencari "Environment Variables" di pengaturan sistem, mengedit "User variables" atau "System variables", dan menambahkan `export OPENAI_API_KEY=kunci_Anda_di_sini`.

Pilihan kami adalah membuat file `config.py` di mana semua kunci API disimpan. Kemudian kami mengimpor fungsi dari modul ini yang memuat kunci‑kunci ini ke dalam variabel lingkungan. Pendekatan ini memusatkan manajemen kredensial dan memudahkan pembaruan kunci saat diperlukan:

```python
import os
OPENAI_API_KEY =  "... "
# Saya menghilangkan semua kunci lainnya
def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
             os.environ[key] = value
```

Jika Anda mencari file ini di repositori GitHub, Anda akan melihatnya tidak ada. Ini disengaja – saya telah mengecualikannya dari pelacakan Git menggunakan file `.gitignore`. File `.gitignore` memberi tahu Git file mana yang harus diabaikan saat melakukan commit perubahan, yang penting untuk:

1. Mencegah kredensial sensitif terekspos secara publik
2. Menghindari commit tidak sengaja kunci API pribadi
3. Melindungi diri Anda dari biaya penggunaan tidak sah

Untuk mengimplementasikannya sendiri, cukup tambahkan `config.py` ke file `.gitignore` Anda:

```
# Di .gitignore
config.py
.env
**/api_keys.txt
# File sensitif lainnya
```

Anda dapat mengatur semua kunci Anda di file `config.py`. Fungsi `set_environment()` ini memuat semua kunci ke dalam lingkungan seperti yang disebutkan. Kapan pun Anda ingin menjalankan aplikasi, Anda mengimpor fungsi dan menjalankannya seperti ini:

```python
from config import set_environment
set_environment()
```

Untuk lingkungan produksi, pertimbangkan untuk menggunakan layanan manajemen rahasia khusus atau variabel lingkungan yang disuntikkan saat runtime. Pendekatan ini memberikan keamanan tambahan sambil mempertahankan pemisahan antara kode dan kredensial.

Meskipun model OpenAI tetap berpengaruh, ekosistem LLM telah dengan cepat terdiversifikasi, menawarkan pengembang banyak opsi untuk aplikasi mereka. Untuk menjaga kejelasan, kami akan memisahkan LLM dari gateway model yang menyediakan akses ke mereka.

- **Keluarga LLM kunci**
  - **Anthropic Claude**: Unggul dalam penalaran, pemrosesan konten panjang, dan analisis visi dengan jendela konteks hingga 200K token
  - **Model Mistral**: Model sumber terbuka yang kuat dengan kemampuan multibahasa yang kuat dan kemampuan penalaran luar biasa
  - **Google Gemini**: Model multimodal canggih dengan jendela konteks 1M token terdepan di industri dan akses informasi real‑time
  - **OpenAI GPT‑o**: Kemampuan omnimodal terdepan menerima teks, audio, gambar, dan video dengan penalaran yang ditingkatkan
  - **Model DeepSeek**: Spesialis dalam pengkodean dan penalaran teknis dengan kinerja terdepan pada tugas pemrograman
  - **AI21 Labs Jurassic**: Kuat dalam aplikasi akademik dan generasi konten panjang
  - **Inflection Pi**: Dioptimalkan untuk AI percakapan dengan kecerdasan emosional luar biasa
  - **Model Perplexity**: Fokus pada jawaban akurat dan dikutip untuk aplikasi penelitian
  - **Model Cohere**: Spesialis untuk aplikasi perusahaan dengan kemampuan multibahasa yang kuat
- **Gateway penyedia cloud**
  - **Amazon Bedrock**: Akses API terpadu ke model dari Anthropic, AI21, Cohere, Mistral, dan lainnya dengan integrasi AWS
  - **Azure OpenAI Service**: Akses tingkat perusahaan ke OpenAI dan model lainnya dengan keamanan yang kuat dan integrasi ekosistem Microsoft
  - **Google Vertex AI**: Akses ke Gemini dan model lainnya dengan integrasi Google Cloud yang mulus
- **Platform independen**
  - **Together AI**: Menghosting 200+ model sumber terbuka dengan opsi GPU tanpa server dan khusus
  - **Replicate**: Spesialis dalam menyebarkan model sumber terbuka multimodal dengan harga bayar‑sesuai‑penggunaan
  - **HuggingFace Inference Endpoints**: Penyebaran produksi ribuan model sumber terbuka dengan kemampuan penyesuaian

Sepanjang buku ini, kami akan bekerja dengan berbagai model yang diakses melalui penyedia berbeda, memberi Anda fleksibilitas untuk memilih opsi terbaik untuk kebutuhan spesifik dan persyaratan infrastruktur Anda.

Kami akan menggunakan OpenAI untuk banyak aplikasi tetapi juga akan mencoba LLM dari organisasi lain.

:::note
Ada dua paket integrasi utama:

- `langchain‑google‑vertexai`
- `langchain‑google‑genai`

Kami akan menggunakan `langchain‑google‑genai`, paket yang direkomendasikan LangChain untuk pengembang individu. Penyiapan jauh lebih sederhana, hanya memerlukan akun Google dan kunci API. Disarankan untuk beralih ke `langchain‑google‑vertexai` untuk proyek yang lebih besar. Integrasi ini menawarkan fitur perusahaan seperti kunci enkripsi pelanggan, integrasi virtual private cloud, dan lainnya, memerlukan akun Google Cloud dengan penagihan.

Jika Anda telah mengikuti instruksi di GitHub, seperti yang ditunjukkan di bagian sebelumnya, Anda seharusnya sudah menginstal paket `langchain‑google‑genai`.
:::

## Menjelajahi blok bangunan LangChain

Untuk membangun aplikasi praktis, kita perlu tahu cara bekerja dengan penyedia model yang berbeda. Mari jelajahi berbagai opsi yang tersedia, dari layanan cloud hingga penyebaran lokal. Kita akan mulai dengan konsep dasar seperti LLM dan model obrolan, lalu menyelami prompt, rantai, dan sistem memori.

## Antarmuka model

LangChain menyediakan antarmuka terpadu untuk bekerja dengan berbagai penyedia LLM. Abstraksi ini memudahkan beralih antara model berbeda sambil mempertahankan struktur kode yang konsisten. Contoh‑contoh berikut menunjukkan cara mengimplementasikan komponen inti LangChain dalam skenario praktis.

:::note
Harap dicatat bahwa pengguna hampir secara eksklusif harus menggunakan model obrolan yang lebih baru karena sebagian besar penyedia model telah mengadopsi antarmuka seperti obrolan untuk berinteraksi dengan model bahasa. Kami masih menyediakan antarmuka LLM, karena sangat mudah digunakan sebagai string‑masuk, string‑keluar.
:::

### Pola interaksi LLM

Antarmuka LLM mewakili model penyelesaian teks tradisional yang mengambil input string dan mengembalikan output string. Semakin banyak kasus penggunaan di LangChain hanya menggunakan antarmuka ChatModel, terutama karena lebih cocok untuk membangun alur kerja kompleks dan mengembangkan agen. Dokumentasi LangChain sekarang menghentikan antarmuka LLM dan merekomendasikan penggunaan antarmuka berbasis obrolan. Meskipun bab ini mendemonstrasikan kedua antarmuka, kami merekomendasikan menggunakan model obrolan karena mereka mewakili standar terkini untuk tetap mutakhir dengan LangChain.

Mari kita lihat antarmuka LLM dalam aksi:

```python
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
# Inisialisasi model OpenAI
openai_llm = OpenAI()
# Inisialisasi model Gemini
gemini_pro = GoogleGenerativeAI(model="gemini-1.5-pro")
# Salah satu atau keduanya dapat digunakan dengan antarmuka yang sama
response = openai_llm.invoke("Ceritakan lelucon tentang bola lampu!")
print(response)
```

Harap dicatat bahwa Anda harus mengatur variabel lingkungan Anda ke kunci penyedia saat menjalankan ini. Misalnya, saat menjalankan ini saya akan memulai file dengan memanggil `set_environment() from config`:

```python
from config import set_environment
set_environment()
```

Kami mendapatkan output ini:

```
Why did the light bulb go to therapy?
Because it was feeling a little dim!
```

Untuk model Gemini, kita dapat menjalankan:

```python
response = gemini_pro.invoke("Ceritakan lelucon tentang bola lampu!")
```

Untuk saya, Gemini menghasilkan lelucon ini:

```
Why did the light bulb get a speeding ticket?
Because it was caught going over the watt limit!
```

Perhatikan bagaimana kami menggunakan metode `invoke()` yang sama terlepas dari penyedia. Konsistensi ini memudahkan bereksperimen dengan model berbeda atau beralih penyedia dalam produksi.

### Pengujian pengembangan

Selama pengembangan, Anda mungkin ingin menguji aplikasi tanpa melakukan panggilan API aktual. LangChain menyediakan `FakeListLLM` untuk tujuan ini:

```python
from langchain_community.llms import FakeListLLM
# Buat LLM palsu yang selalu mengembalikan respons yang sama
fake_llm = FakeListLLM(responses=["Hello"])
result = fake_llm.invoke("Input apa pun akan mengembalikan Hello")
print(result)  # Output: Hello
```

### Bekerja dengan model obrolan

Model obrolan adalah LLM yang disesuaikan untuk interaksi multi‑giliran antara model dan manusia. Saat ini sebagian besar LLM disesuaikan untuk percakapan multi‑giliran. Alih‑alih memberikan input ke model, seperti:

```
human: giliran1
ai: jawaban1
human: giliran2
ai: jawaban2
```

di mana kita mengharapkannya menghasilkan output dengan melanjutkan percakapan, saat ini penyedia model biasanya mengekspos API yang mengharapkan setiap giliran sebagai bagian payload yang terformat baik secara terpisah. Penyedia model biasanya tidak menyimpan riwayat obrolan di sisi server, mereka mendapatkan riwayat lengkap dikirim setiap kali dari klien dan hanya memformat prompt akhir di sisi server.

LangChain mengikuti pola yang sama dengan ChatModels, memproses percakapan melalui pesan terstruktur dengan peran dan konten. Setiap pesan berisi:

- Peran (siapa yang berbicara), yang didefinisikan oleh kelas pesan (semua pesan mewarisi dari BaseMessage)
- Konten (apa yang dikatakan)

Jenis pesan meliputi:

- `SystemMessage`: Mengatur perilaku dan konteks untuk model. Contoh:
  `SystemMessage(content="Anda adalah asisten pemrograman yang membantu")`
- `HumanMessage`: Mewakili input pengguna seperti pertanyaan, perintah, dan data. Contoh:
  `HumanMessage(content="Tulis fungsi Python untuk menghitung faktorial")`
- `AIMessage`: Berisi respons model

Mari kita lihat ini dalam aksi:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
chat = ChatAnthropic(model="claude-3-opus-20240229")
messages = [
    SystemMessage(content="Anda adalah asisten pemrograman yang membantu"),
    HumanMessage(content="Tulis fungsi Python untuk menghitung faktorial")
]
response = chat.invoke(messages)
print(response)
```

Claude menghasilkan fungsi, penjelasan, dan contoh untuk memanggil fungsi.

Berikut fungsi Python yang menghitung faktorial dari angka yang diberikan:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Faktorial tidak terdefinisi untuk angka negatif.")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
```

Mari kita uraikan. Fungsi `factorial` dirancang untuk mengambil integer `n` sebagai input dan menghitung faktorialnya. Ini dimulai dengan memeriksa apakah `n` negatif, dan jika ya, memunculkan `ValueError` karena faktorial tidak terdefinisi untuk angka negatif. Jika `n` adalah nol, fungsi mengembalikan `1`, yang masuk akal karena, menurut definisi, faktorial 0 adalah 1.
Saat berurusan dengan angka positif, fungsi memulai dengan mengatur variabel `result` ke 1. Dari sana, memasuki loop yang berjalan dari 1 hingga `n`, inklusif, berkat fungsi `range`. Selama setiap langkah loop, ia mengalikan hasil dengan angka saat ini, secara bertahap membangun faktorial. Setelah loop selesai, fungsi mengembalikan nilai yang dihitung akhir. Anda dapat memanggil fungsi ini dengan memberikan integer non‑negatif sebagai argumen. Berikut beberapa contoh:

```python
print(factorial(0))  # Output: 1
print(factorial(5))  # Output: 120
print(factorial(10))  # Output: 3628800
print(factorial(-5))  # Memunculkan ValueError: Faktorial tidak terdefinisi untuk angka negatif.
```

Perhatikan bahwa fungsi faktorial tumbuh sangat cepat, sehingga menghitung faktorial angka besar dapat melebihi nilai maksimum yang dapat direpresentasikan dalam Python. Dalam kasus seperti itu, Anda mungkin perlu menggunakan pendekatan berbeda atau pustaka yang mendukung aritmatika presisi arbitrer.

Demikian pula, kita bisa bertanya kepada model OpenAI seperti GPT‑4 atau GPT‑4o:

```python
from langchain_openai.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name='gpt-4o')
```

### Model penalaran

Claude 3.7 Sonnet Anthropic memperkenalkan kemampuan kuat yang disebut _pemikiran diperluas_ yang memungkinkan model menunjukkan proses penalarannya sebelum memberikan jawaban akhir. Fitur ini mewakili kemajuan signifikan dalam cara pengembang memanfaatkan LLM untuk tugas penalaran kompleks.

Berikut cara mengonfigurasi pemikiran diperluas melalui kelas ChatAnthropic:

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
# Buat template
template = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah programmer berpengalaman dan analis matematika."),
    ("user", "{problem}")
])
# Inisialisasi Claude dengan pemikiran diperluas diaktifkan
chat = ChatAnthropic(
    model_name="claude-3-7-sonnet-20240326",  # Gunakan versi model terbaru
    max_tokens=64_000,                        # Batas panjang respons total
    thinking={"type": "enabled", "budget_tokens": 15000},  # Alokasikan token untuk berpikir
)
# Buat dan jalankan rantai
chain = template | chat
# Masalah algoritmik kompleks
problem = """
Rancang algoritma untuk menemukan elemen terbesar ke‑k dalam array tidak terurut
dengan kompleksitas waktu optimal. Analisis kompleksitas waktu dan ruang
solusi Anda dan jelaskan mengapa itu optimal.
"""
# Dapatkan respons dengan pemikiran disertakan
response = chat.invoke([HumanMessage(content=problem)])
print(response.content)
```

Respons akan mencakup penalaran langkah‑demi‑langkah Claude tentang pemilihan algoritma, analisis kompleksitas, dan pertimbangan optimasi sebelum menyajikan solusi akhir. Dalam contoh sebelumnya:

- Dari 64.000 token maksimum panjang respons, hingga 15.000 token dapat digunakan untuk proses berpikir Claude.
- Sekitar 49.000 token tersisa tersedia untuk respons akhir.
- Claude tidak selalu menggunakan seluruh anggaran berpikir—ia menggunakan apa yang dibutuhkan untuk tugas spesifik. Jika Claude kehabisan token berpikir, ia akan beralih ke jawaban akhir.

Sementara Claude menawarkan konfigurasi berpikir eksplisit, Anda dapat mencapai hasil serupa (meski tidak identik) dengan penyedia lain melalui teknik berbeda:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah asisten pemecahan masalah."),
    ("user", "{problem}")
])
# Inisialisasi dengan parameter reasoning_effort
chat = ChatOpenAI(
    model="o3-mini",
    reasoning_effort="high"  # Opsi: "low", "medium", "high"
)
chain = template | chat
response = chain.invoke({"problem": "Hitung strategi optimal untuk..."})
chat = ChatOpenAI(model="gpt-4o")
chain = template | chat
response = chain.invoke({"problem": "Hitung strategi optimal untuk..."})
```

Parameter `reasoning_effort` menyederhanakan alur kerja Anda dengan menghilangkan kebutuhan akan prompt penalaran kompleks, memungkinkan Anda menyesuaikan kinerja dengan mengurangi upaya ketika kecepatan lebih penting daripada analisis terperinci, dan membantu mengelola konsumsi token dengan mengontrol berapa banyak daya pemrosesan yang digunakan untuk proses penalaran.

Model DeepSeek juga menawarkan konfigurasi berpikir eksplisit melalui integrasi LangChain.

### Mengontrol perilaku model

Memahami cara mengontrol perilaku LLM sangat penting untuk menyesuaikan outputnya dengan kebutuhan spesifik. Tanpa penyesuaian parameter yang cermat, model mungkin menghasilkan respons yang terlalu kreatif, tidak konsisten, atau verbose yang tidak cocok untuk aplikasi praktis. Misalnya, dalam layanan pelanggan, Anda menginginkan jawaban yang konsisten dan faktual, sementara dalam generasi konten, Anda mungkin menargetkan output yang lebih kreatif dan promosional.

LLM menawarkan beberapa parameter yang memungkinkan kontrol halus atas perilaku generasi, meskipun implementasi pasti dapat bervariasi antar penyedia. Mari jelajahi yang paling penting:

| Parameter                       | Deskripsi                                                                   | Rentang Khas                                 | Terbaik Untuk                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Temperature**                 | Mengontrol keacakan dalam generasi teks                                     | 0.0‑1.0 (OpenAI, Anthropic) 0.0‑2.0 (Gemini) | Lebih rendah (0.0‑0.3): Tugas faktual, T&J Lebih tinggi (0.7+): Penulisan kreatif, brainstorming    |
| **Top‑k**                       | Membatasi pemilihan token ke k token paling mungkin                         | 1‑100                                        | Nilai lebih rendah (1‑10): Output lebih terfokus Nilai lebih tinggi: Penyelesaian lebih beragam     |
| **Top‑p (Sampling Nukleus)**    | Mempertimbangkan token hingga probabilitas kumulatif mencapai ambang batas  | 0.0‑1.0                                      | Nilai lebih rendah (0.5): Output lebih terfokus Nilai lebih tinggi (0.9): Respons lebih eksploratif |
| **Max tokens**                  | Membatasi panjang respons maksimum                                          | Spesifik model                               | Mengontrol biaya dan mencegah output verbose                                                        |
| **Penalti kehadiran/frekuensi** | Mencegah pengulangan dengan memberikan penalti pada token yang telah muncul | ‑2.0 hingga 2.0                              | Generasi konten panjang di mana pengulangan tidak diinginkan                                        |
| **Urutan berhenti**             | Memberi tahu model kapan harus berhenti menghasilkan                        | String kustom                                | Mengontrol titik akhir persis dari generasi                                                         |

Tabel 2.2: Parameter yang ditawarkan oleh LLM

Parameter‑parameter ini bekerja bersama untuk membentuk output model:

- **Temperature + Top‑k/Top‑p**: Pertama, Top‑k/Top‑p menyaring distribusi token, lalu temperature memengaruhi keacakan dalam set yang disaring
- **Penalti + Temperature**: Temperature lebih tinggi dengan penalti rendah dapat menghasilkan teks kreatif tetapi berpotensi berulang

LangChain menyediakan antarmuka konsisten untuk mengatur parameter ini di berbagai penyedia LLM:

```python
from langchain_openai import OpenAI
# Untuk respons faktual dan konsisten
factual_llm = OpenAI(temperature=0.1, max_tokens=256)
# Untuk brainstorming kreatif
creative_llm = OpenAI(temperature=0.8, top_p=0.95, max_tokens=512)
```

Beberapa pertimbangan spesifik penyedia yang perlu diingat adalah:

- **OpenAI**: Dikenal dengan perilaku konsisten dengan temperature dalam rentang 0.0‑1.0
- **Anthropic**: Mungkin perlu pengaturan temperature lebih rendah untuk mencapai tingkat kreativitas serupa dengan penyedia lain
- **Gemini**: Mendukung temperature hingga 2.0, memungkinkan kreativitas lebih ekstrem pada pengaturan lebih tinggi
- **Model sumber terbuka**: Sering memerlukan kombinasi parameter berbeda daripada API komersial

### Memilih parameter untuk aplikasi

Untuk aplikasi perusahaan yang memerlukan konsistensi dan akurasi, temperature lebih rendah (0.0‑0.3) dikombinasikan dengan nilai top‑p moderat (0.5‑0.7) biasanya lebih disukai. Untuk asisten kreatif atau alat brainstorming, temperature lebih tinggi menghasilkan output lebih beragam, terutama ketika dipasangkan dengan nilai top‑p lebih tinggi.

Ingatlah bahwa penyetelan parameter seringkali empiris – mulai dengan rekomendasi penyedia, lalu sesuaikan berdasarkan kebutuhan aplikasi spesifik dan output yang diamati.

## Prompt dan template

Rekayasa prompt adalah keterampilan penting untuk pengembangan aplikasi LLM, terutama di lingkungan produksi. LangChain menyediakan sistem yang kokoh untuk mengelola prompt dengan fitur‑fitur yang mengatasi tantangan pengembangan umum:

- **Sistem template** untuk generasi prompt dinamis
- **Manajemen dan versi prompt** untuk melacak perubahan
- **Manajemen contoh sedikit‑shot** untuk meningkatkan kinerja model
- **Penguraian dan validasi output** untuk hasil yang andal

Template prompt LangChain mengubah teks statis menjadi prompt dinamis dengan substitusi variabel – bandingkan dua pendekatan ini untuk melihat perbedaan kunci:

1. Penggunaan statis – bermasalah pada skala:
   ```python
    def generate_prompt(question, context=None):
       if context:
           return f"Informasi konteks: {context}\n\nJawab pertanyaan ini dengan singkat: {question}"
       return f"Jawab pertanyaan ini dengan singkat: {question}"
      # contoh penggunaan:
      prompt_text = generate_prompt("Apa ibu kota Prancis?")
   ```
2. PromptTemplate – siap produksi:
   ```python
   from langchain_core.prompts import PromptTemplate
   # Tentukan sekali, gunakan di mana saja
   question_template = PromptTemplate.from_template( "Jawab pertanyaan ini dengan singkat: {question}" )
   question_with_context_template = PromptTemplate.from_template( "Informasi konteks: {context}\n\nJawab pertanyaan ini dengan singkat: {question}" )
   # Hasilkan prompt dengan mengisi variabel
   prompt_text = question_template.format(question="Apa ibu kota Prancis?")
   ```

Template penting – inilah alasannya:

- **Konsistensi**: Mereka menstandarkan prompt di seluruh aplikasi Anda.
- **Pemeliharaan**: Mereka memungkinkan Anda mengubah struktur prompt di satu tempat alih‑alih di seluruh basis kode Anda.
- **Keterbacaan**: Mereka dengan jelas memisahkan logika template dari logika bisnis.
- **Dapat diuji**: Lebih mudah untuk menguji unit generasi prompt secara terpisah dari panggilan LLM.

Dalam aplikasi produksi, Anda sering perlu mengelola puluhan atau ratusan prompt. Template memberikan cara yang dapat diskalakan untuk mengatur kompleksitas ini.

### Template prompt obrolan

Untuk model obrolan, kita dapat membuat prompt lebih terstruktur yang menggabungkan peran berbeda:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
template = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah penerjemah Inggris ke Prancis."),
    ("user", "Terjemahkan ini ke Prancis: {text}")
])
chat = ChatOpenAI()
formatted_messages = template.format_messages(text="Halo, apa kabar?")
response = chat.invoke(formatted_messages)
print(response)
```

Mari kita mulai dengan melihat **Bahasa Ekspresi LangChain** (**LCEL**), yang memberikan cara bersih dan intuitif untuk membangun aplikasi LLM.

## Bahasa Ekspresi LangChain (LCEL)

LCEL mewakili evolusi signifikan dalam cara kita membangun aplikasi bertenaga LLM dengan LangChain. Diperkenalkan pada Agustus 2023, LCEL adalah pendekatan deklaratif untuk menyusun alur kerja LLM kompleks. Alih‑alih fokus pada _bagaimana_ mengeksekusi setiap langkah, LCEL memungkinkan Anda mendefinisikan _apa_ yang ingin Anda capai, memungkinkan LangChain menangani detail eksekusi di belakang layar.

Pada intinya, LCEL berfungsi sebagai lapisan kode minimalis yang membuatnya sangat mudah menghubungkan komponen LangChain berbeda. Jika Anda terbiasa dengan pipa Unix atau pustaka pemrosesan data seperti pandas, Anda akan mengenali sintaks intuitif: komponen dihubungkan menggunakan operator pipa (|) untuk membuat pipa pemrosesan.

Seperti yang kami perkenalkan secara singkat di [Bab 1](Chapter_1.xhtml#_idTextAnchor000), LangChain selalu menggunakan konsep "rantai" sebagai pola fundamental untuk menghubungkan komponen. Rantai mewakili urutan operasi yang mengubah input menjadi output.

Awalnya, LangChain mengimplementasikan pola ini melalui kelas `Chain` spesifik seperti `LLMChain` dan `ConversationChain`. Meskipun kelas warisan ini masih ada, mereka telah dihentikan demi pendekatan LCEL yang lebih fleksibel dan kuat, yang dibangun di atas antarmuka Runnable.

Antarmuka Runnable adalah landasan LangChain modern. Runnable adalah komponen apa pun yang dapat memproses input dan menghasilkan output dengan cara terstandarisasi. Setiap komponen yang dibangun dengan LCEL mematuhi antarmuka ini, yang menyediakan metode konsisten termasuk:

- `invoke()`: Memproses satu input secara sinkron dan mengembalikan output
- `stream()`: Streaming output saat sedang dihasilkan
- `batch()`: Memproses beberapa input secara paralel dengan efisien
- `ainvoke()`, `abatch()`, `astream()`: Versi asinkron dari metode di atas

Standarisasi ini berarti setiap komponen Runnable—baik itu LLM, template prompt, pengambil dokumen, atau fungsi kustom—dapat dihubungkan ke Runnable lainnya, menciptakan sistem komposabilitas yang kuat.

Setiap Runnable mengimplementasikan serangkaian metode konsisten termasuk:

- `invoke()`: Memproses satu input secara sinkron dan mengembalikan output
- `stream()`: Streaming output saat sedang dihasilkan

Standarisasi ini kuat karena berarti setiap komponen Runnable—baik itu LLM, template prompt, pengambil dokumen, atau fungsi kustom—dapat dihubungkan ke Runnable lainnya. Konsistensi antarmuka ini memungkinkan aplikasi kompleks dibangun dari blok bangunan yang lebih sederhana.

:::note
LCEL menawarkan beberapa keunggulan yang menjadikannya pendekatan pilihan untuk membangun aplikasi LangChain:

- **Pengembangan cepat**: Sintaks deklaratif memungkinkan prototipe dan iterasi rantai kompleks lebih cepat.
- **Fitur siap produksi**: LCEL menyediakan dukungan bawaan untuk streaming, eksekusi asinkron, dan pemrosesan paralel.
- **Keterbacaan yang ditingkatkan**: Sintaks pipa memudahkan memvisualisasikan aliran data melalui aplikasi Anda.
- **Integrasi ekosistem mulus**: Aplikasi yang dibangun dengan LCEL otomatis bekerja dengan LangSmith untuk observabilitas dan LangServe untuk penyebaran.
- **Dapat disesuaikan**: Sertakan fungsi Python kustom dengan mudah ke dalam rantai Anda dengan RunnableLambda.
- **Optimasi runtime**: LangChain dapat secara otomatis mengoptimalkan eksekusi rantai yang didefinisikan LCEL.
  :::

LCEL benar‑benar bersinar ketika Anda perlu membangun aplikasi kompleks yang menggabungkan banyak komponen dalam alur kerja canggih. Di bagian selanjutnya, kita akan menjelajahi cara menggunakan LCEL untuk membangun aplikasi dunia nyata, mulai dari blok bangunan dasar dan secara bertahap memasukkan pola yang lebih canggih.

Operator pipa (|) berfungsi sebagai landasan LCEL, memungkinkan Anda merantai komponen secara berurutan:

```python
# 1. Rantai berurutan dasar: Hanya prompt ke LLM
basic_chain = prompt | llm | StrOutputParser()
```

Di sini, `StrOutputParser()` adalah pengurai output sederhana yang mengekstrak respons string dari LLM. Ini mengambil output terstruktur dari LLM dan mengubahnya menjadi string biasa, membuatnya lebih mudah digunakan. Pengurai ini terutama berguna ketika Anda hanya membutuhkan konten teks tanpa metadata.

Di balik layar, LCEL menggunakan operator overloading Python untuk mengubah ekspresi ini menjadi RunnableSequence di mana output setiap komponen mengalir ke input komponen berikutnya. Pipa (|) adalah syntactic sugar yang mengganti metode tersembunyi `__or__`, dengan kata lain, `A | B` setara dengan `B.__or__(A)`.

Sintaks pipa setara dengan membuat `RunnableSequence` secara terprogram:

```
chain = RunnableSequence(first= prompt, middle=[llm], last= output_parser)
LCEL juga mendukung penambahan transformasi dan fungsi kustom:
with_transformation = prompt | llm | (lambda x: x.upper()) | StrOutputParser()
```

Untuk alur kerja lebih kompleks, Anda dapat menggabungkan logika percabangan:

```
decision_chain = prompt | llm | (lambda x: route_based_on_content(x)) | {
    "summarize": summarize_chain,
    "analyze": analyze_chain
}
```

Elemen non‑Runnable seperti fungsi dan kamus secara otomatis diubah ke tipe Runnable yang sesuai:

```python
# Fungsi ke Runnable
length_func = lambda x: len(x)
chain = prompt | length_func | output_parser
# Dikonversi menjadi:
chain = prompt | RunnableLambda(length_func) | output_parser
```

Sifat LCEL yang fleksibel dan dapat disusun akan memungkinkan kita menangani tantangan aplikasi LLM dunia nyata dengan kode yang elegan dan mudah dipelihara.

### Alur kerja sederhana dengan LCEL

Seperti yang telah kita lihat, LCEL menyediakan sintaks deklaratif untuk menyusun komponen aplikasi LLM menggunakan operator pipa. Pendekatan ini sangat menyederhanakan konstruksi alur kerja dibandingkan kode imperatif tradisional. Mari bangun generator lelucon sederhana untuk melihat LCEL dalam aksi:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# Buat komponen
prompt = PromptTemplate.from_template("Ceritakan lelucon tentang {topic}")
llm = ChatOpenAI()
output_parser = StrOutputParser()
# Rantai bersama menggunakan LCEL
chain = prompt | llm | output_parser
#  Eksekusi alur kerja dengan satu panggilan
result = chain.invoke({"topic": "pemrograman"})
print(result)
```

Ini menghasilkan lelucon pemrograman:

```
Why don't programmers like nature?
It has too many bugs!
```

Tanpa LCEL, alur kerja yang sama setara dengan panggilan fungsi terpisah dengan penerusan data manual:

```python
formatted_prompt = prompt.invoke({"topic": "pemrograman"})
llm_output = llm.invoke(formatted_prompt)
result = output_parser.invoke(llm_output)
```

Seperti yang Anda lihat, kami telah memisahkan konstruksi rantai dari eksekusinya.

Dalam aplikasi produksi, pola ini menjadi lebih berharga saat menangani alur kerja kompleks dengan logika percabangan, penanganan kesalahan, atau pemrosesan paralel – topik yang akan kita jelajahi di [Bab 3](Chapter_3.xhtml#_idTextAnchor049).

### Contoh rantai kompleks

Sementara generator lelucon sederhana menunjukkan penggunaan LCEL dasar, aplikasi dunia nyata biasanya memerlukan penanganan data yang lebih canggih. Mari jelajahi pola lanjutan menggunakan contoh generasi dan analisis cerita.

Dalam contoh ini, kita akan membangun alur kerja multi‑tahap yang menunjukkan cara:

1. Menghasilkan konten dengan satu panggilan LLM
2. Memberikan konten itu ke panggilan LLM kedua
3. Melestarikan dan mengubah data di seluruh rantai

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
# Inisialisasi model
llm = GoogleGenerativeAI(model="gemini-1.5-pro")
# Rantai pertama menghasilkan cerita
story_prompt = PromptTemplate.from_template("Tulis cerita pendek tentang {topic}")
story_chain = story_prompt | llm | StrOutputParser()
# Rantai kedua menganalisis cerita
analysis_prompt = PromptTemplate.from_template(
    "Analisis suasana cerita berikut:\n{story}"
)
analysis_chain = analysis_prompt | llm | StrOutputParser()
```

Kita dapat menyusun kedua rantai ini bersama. Pendekatan sederhana pertama kami memipakan cerita langsung ke rantai analisis:

```python
# Gabungkan rantai
story_with_analysis = story_chain | analysis_chain
# Jalankan rantai gabungan
story_analysis = story_with_analysis.invoke({"topic": "hari hujan"})
print("\nAnalisis:", story_analysis)
```

Saya mendapatkan analisis panjang. Inilah awalnya:

```
Analisis: Suasana cerita didominasi **tenang, damai, dan sedikit romantis.** Ada rasa melankoli lembut yang dibawa oleh hujan dan kehampaan toko buku yang sepi, tetapi ini diseimbangkan oleh perasaan hangat dan harapan.
```

Meskipun ini berhasil, kita kehilangan cerita asli dalam hasil kita – kita hanya mendapatkan analisis! Dalam aplikasi produksi, kami biasanya ingin melestarikan konteks di seluruh rantai:

```python
from langchain_core.runnables import RunnablePassthrough
# Menggunakan RunnablePassthrough.assign untuk melestarikan data
enhanced_chain = RunnablePassthrough.assign(
    story=story_chain  # Tambahkan kunci 'story' dengan konten yang dihasilkan
).assign(
    analysis=analysis_chain  # Tambahkan kunci 'analysis' dengan analisis cerita
)
# Eksekusi rantai
result = enhanced_chain.invoke({"topic": "hari hujan"})
print(result.keys())  # Output: dict_keys(['topic', 'story', 'analysis'])  # dict_keys(['topic', 'story', 'analysis'])
```

Untuk kontrol lebih atas struktur output, kita juga dapat membuat kamus secara manual:

```python
from operator import itemgetter
# Pendekatan alternatif menggunakan konstruksi kamus
manual_chain = (
    RunnablePassthrough() |  # Lewatkan input
    {
        "story": story_chain,  # Tambahkan hasil cerita
        "topic": itemgetter("topic")  # Pertahankan topik asli
    } |
    RunnablePassthrough().assign(  # Tambahkan analisis berdasarkan cerita
        analysis=analysis_chain
    )
)
result = manual_chain.invoke({"topic": "hari hujan"})
print(result.keys())  # Output: dict_keys(['story', 'topic', 'analysis'])
```

Kita dapat menyederhanakan ini dengan konversi kamus menggunakan singkatan LCEL:

```python
# Konstruksi kamus yang disederhanakan
simple_dict_chain_corrected = story_chain | {
    "story": RunnablePassthrough(),  # Lewatkan output cerita sebagai 'story'
    "analysis": analysis_chain
}
# analysis_chain akan menerima {'story': 'konten cerita aktual'} seperti yang diharapkan.
result_corrected = simple_dict_chain_corrected.invoke({"topic": "hari hujan"})
print(result_corrected.keys())
```

Apa yang membuat contoh‑contoh ini lebih kompleks daripada generator lelucon sederhana kami?

- **Beberapa panggilan LLM**: Alih‑alih aliran prompt → LLM → pengurai tunggal, kami merantai beberapa interaksi LLM
- **Transformasi data**: Menggunakan alat seperti `RunnablePassthrough` dan `itemgetter` untuk mengelola dan mengubah data
- **Pelestarian kamus**: Mempertahankan konteks di seluruh rantai alih‑alih hanya meneruskan nilai tunggal
- **Output terstruktur**: Membuat kamus output terstruktur alih‑alih string sederhana

Pola‑pola ini penting untuk aplikasi produksi di mana Anda perlu:

- Melacak asal konten yang dihasilkan
- Menggabungkan hasil dari beberapa operasi
- Struktur data untuk pemrosesan atau tampilan hilir
- Menerapkan penanganan kesalahan yang lebih canggih

:::note
Sementara LCEL menangani banyak alur kerja kompleks dengan elegan, untuk manajemen status dan logika percabangan lanjutan, Anda akan ingin menjelajahi LangGraph, yang akan kita bahas di [Bab 3](Chapter_3.xhtml#_idTextAnchor049).
:::

Sementara contoh sebelumnya kami menggunakan model berbasis cloud seperti OpenAI dan Gemini Google, LCEL LangChain dan fungsionalitas lainnya bekerja mulus dengan model lokal juga. Fleksibilitas ini memungkinkan Anda memilih pendekatan penyebaran yang tepat untuk kebutuhan spesifik Anda.

## Menjalankan model lokal

Saat membangun aplikasi LLM dengan LangChain, Anda perlu memutuskan di mana model Anda akan berjalan.

- Keunggulan model lokal:
  - Kontrol data lengkap dan privasi
  - Tidak ada biaya API atau batasan penggunaan
  - Tidak bergantung pada internet
  - Kontrol atas parameter model dan penyesuaian
- Keunggulan model cloud:
  - Tidak ada persyaratan perangkat keras atau kompleksitas setup
  - Akses ke model paling kuat dan mutakhir
  - Skalabilitas elastis tanpa manajemen infrastruktur
  - Perbaikan model berkelanjutan tanpa pembaruan manual
- Kapan memilih model lokal:
  - Aplikasi dengan persyaratan privasi data ketat
  - Lingkungan pengembangan dan pengujian
  - Skenario penyebaran tepi atau offline
  - Aplikasi sensitif biaya dengan penggunaan volume tinggi yang dapat diprediksi

Mari kita mulai dengan salah satu opsi paling ramah pengembang untuk menjalankan model lokal.

## Memulai dengan Ollama

Ollama menyediakan cara ramah pengembang untuk menjalankan model sumber terbuka yang kuat secara lokal. Ini memberikan antarmuka sederhana untuk mengunduh dan menjalankan berbagai model sumber terbuka. Dependensi `langchain‑ollama` seharusnya sudah diinstal jika Anda telah mengikuti instruksi di bab ini; namun, mari kita bahas secara singkat:

1. Instal integrasi LangChain Ollama:
   ```
   pip install langchain‑ollama
   ```
2. Kemudian tarik model. Dari baris perintah, terminal seperti bash atau WindowsPowerShell, jalankan:
   ```
   ollama pull deepseek‑r1:1.5b
   ```
3. Mulai server Ollama:
   ```
   ollama serve
   ```

Berikut cara mengintegrasikan Ollama dengan pola LCEL yang telah kita jelajahi:

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Inisialisasi Ollama dengan model pilihan Anda
local_llm = ChatOllama(
    model="deepseek‑r1:1.5b",
    temperature=0,
)
# Buat rantai LCEL menggunakan model lokal
prompt = PromptTemplate.from_template("Jelaskan {concept} dalam istilah sederhana")
local_chain = prompt | local_llm | StrOutputParser()
# Gunakan rantai dengan model lokal Anda
result = local_chain.invoke({"concept": "komputasi kuantum"})
print(result)
```

Rantai LCEL ini berfungsi identik dengan contoh berbasis cloud kami, menunjukkan desain agnostik‑model LangChain.

Harap dicatat bahwa karena Anda menjalankan model lokal, Anda tidak perlu menyiapkan kunci apa pun. Jawabannya sangat panjang – meskipun cukup masuk akal. Anda dapat menjalankan ini sendiri dan melihat jawaban apa yang Anda dapatkan.

Setelah kita melihat generasi teks dasar, mari lihat integrasi lain. Hugging Face menawarkan cara yang mudah diakses untuk menjalankan model lokal, dengan akses ke ekosistem besar model yang telah dilatih sebelumnya.

## Bekerja dengan model Hugging Face secara lokal

Dengan Hugging Face, Anda dapat menjalankan model secara lokal (HuggingFacePipeline) atau di Hugging Face Hub (HuggingFaceEndpoint). Di sini, kita berbicara tentang proses lokal, jadi kami akan fokus pada `HuggingFacePipeline`. Ini dia:

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# Buat pipeline dengan model kecil:
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama‑1.1B‑Chat‑v1.0",
    task="text‑generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
chat_model = ChatHuggingFace(llm=llm)
# Gunakan seperti LLM LangChain lainnya
messages = [
    SystemMessage(content="Anda adalah asisten yang membantu"),
    HumanMessage(
        content="Jelaskan konsep pembelajaran mesin dalam istilah sederhana"
    ),
]
ai_msg = chat_model.invoke(messages)
print(ai_msg.content)
```

Ini bisa memakan waktu cukup lama, terutama pertama kali, karena model harus diunduh terlebih dahulu. Kami telah menghilangkan respons model demi singkatnya.

LangChain mendukung menjalankan model lokal melalui integrasi lain juga, misalnya:

- **llama.cpp:** Implementasi C++ berkinerja tinggi ini memungkinkan menjalankan model berbasis LLaMA secara efisien pada perangkat keras konsumen. Meskipun kami tidak akan membahas proses setup secara detail, LangChain menyediakan integrasi langsung dengan llama.cpp untuk inferensi dan penyesuaian.
- **GPT4All** GPT4All menawarkan model ringan yang dapat berjalan pada perangkat keras konsumen. Integrasi LangChain memudahkan penggunaan model ini sebagai pengganti drop‑in untuk LLM berbasis cloud di banyak aplikasi.

Saat Anda mulai bekerja dengan model lokal, Anda akan ingin mengoptimalkan kinerjanya dan menangani tantangan umum. Berikut adalah beberapa tip dan pola penting yang akan membantu Anda memaksimalkan penyebaran lokal Anda dengan LangChain.

## Tips untuk model lokal

Saat bekerja dengan model lokal, ingat poin‑poin ini:

1. **Manajemen sumber daya**: Model lokal memerlukan konfigurasi hati‑hati untuk menyeimbangkan kinerja dan penggunaan sumber daya. Contoh berikut menunjukkan cara mengonfigurasi model Ollama untuk operasi yang efisien:
   ```python
   #  Konfigurasi model dengan pengaturan memori dan pemrosesan yang dioptimalkan
   from langchain_ollama import ChatOllama
   llm = ChatOllama(
     model="mistral:q4_K_M", # model terkuantisasi 4‑bit (jejak memori lebih kecil)
     num_gpu=1, # Jumlah GPU yang digunakan (sesuaikan berdasarkan perangkat keras)
    num_thread=4 # Jumlah thread CPU untuk pemrosesan paralel
   )
   ```

Mari kita lihat apa yang dilakukan setiap parameter:

- **model="mistral:q4_K_M"**: Menentukan versi terkuantisasi 4‑bit dari model Mistral. Kuantisasi mengurangi ukuran model dengan mewakili bobot dengan bit lebih sedikit, mengorbankan presisi minimal untuk penghematan memori signifikan. Misalnya:
  - Model presisi penuh: ~8GB RAM diperlukan
  - Model terkuantisasi 4‑bit: ~2GB RAM diperlukan
- **num_gpu=1**: Mengalokasikan sumber daya GPU. Opsi termasuk:
  - 0: Mode hanya‑CPU (lebih lambat tetapi bekerja tanpa GPU)
  - 1: Menggunakan GPU tunggal (cocok untuk sebagian besar setup desktop)
  - Nilai lebih tinggi: Hanya untuk sistem multi‑GPU
- **num_thread=4**: Mengontrol paralelisasi CPU:
  - Nilai lebih rendah (2‑4): Baik untuk menjalankan bersama aplikasi lain
  - Nilai lebih tinggi (8‑16): Memaksimalkan kinerja pada server khusus
  - Pengaturan optimal: Biasanya cocok dengan jumlah inti fisik CPU Anda

2. **Penanganan kesalahan**: Model lokal dapat mengalami berbagai kesalahan, dari kondisi kehabisan memori hingga penghentian tak terduga. Strategi penanganan kesalahan yang kuat sangat penting:

```python
def safe_model_call(llm, prompt, max_retries=2):
    """Panggil model lokal dengan aman dengan logika coba ulang dan kegagalan anggun"""
    retries = 0
    while retries <= max_retries:
        try:
            return llm.invoke(prompt)
        except RuntimeError as e:
            # Kesalahan umum dengan model lokal saat kehabisan VRAM
            if "CUDA out of memory" in str(e):
                print(f"Kesalahan memori GPU, menunggu dan mencoba ulang ({retries+1}/{max_retries+1})")
                time.sleep(2)  # Beri waktu sistem untuk membebaskan sumber daya
                retries += 1
            else:
                print(f"Kesalahan runtime: {e}")
                return "Terjadi kesalahan saat memproses permintaan Anda."
        except Exception as e:
            print(f"Kesalahan tak terduga memanggil model: {e}")
            return "Terjadi kesalahan saat memproses permintaan Anda."
    # Jika kita kehabisan percobaan ulang
    return "Model saat ini mengalami beban tinggi. Silakan coba lagi nanti."
# Gunakan pembungkus keamanan di rantai LCEL Anda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
prompt = PromptTemplate.from_template("Jelaskan {concept} dalam istilah sederhana")
safe_llm = RunnableLambda(lambda x: safe_model_call(llm, x))
safe_chain = prompt | safe_llm
response = safe_chain.invoke({"concept": "komputasi kuantum"})
```

Kesalahan model lokal umum yang mungkin Anda temui adalah sebagai berikut:

- **Kehabisan memori**: Terjadi ketika model memerlukan lebih banyak VRAM daripada yang tersedia
- **Kegagalan pemuatan model**: Ketika file model rusak atau tidak kompatibel
- **Masalah waktu habis**: Ketika inferensi memakan waktu terlalu lama pada sistem dengan sumber daya terbatas
- **Kesalahan panjang konteks**: Ketika input melebihi batas token maksimum model

Dengan menerapkan optimasi dan strategi penanganan kesalahan ini, Anda dapat membuat aplikasi LangChain yang kokoh yang memanfaatkan model lokal secara efektif sambil mempertahankan pengalaman pengguna yang baik bahkan ketika masalah muncul.

![Gambar 2.1: Bagan keputusan untuk memilih antara model lokal dan berbasis cloud](Images/B32363_02_01.png)

Setelah menjelajahi cara membangun aplikasi berbasis teks dengan LangChain, kami sekarang akan memperluas pemahaman kami ke kemampuan multimodal. Karena sistem AI semakin bekerja dengan berbagai bentuk data, LangChain menyediakan antarmuka untuk menghasilkan gambar dari teks dan memahami konten visual – kemampuan yang melengkapi pemrosesan teks yang telah kami bahas dan membuka kemungkinan baru untuk aplikasi yang lebih imersif.

## Aplikasi AI multimodal

Sistem AI telah berevolusi melampaui pemrosesan hanya‑teks untuk bekerja dengan berbagai tipe data. Dalam lanskap saat ini, kita dapat membedakan antara dua kemampuan kunci yang sering membingungkan tetapi mewakili pendekatan teknologi berbeda.

Pemahaman multimodal mewakili kemampuan model untuk memproses beberapa jenis input secara bersamaan untuk melakukan penalaran dan menghasilkan respons. Sistem canggih ini dapat memahami hubungan antara modalitas berbeda, menerima input seperti teks, gambar, PDF, audio, video, dan data terstruktur. Kemampuan pemrosesan mereka termasuk penalaran lintas‑modal, kesadaran konteks, dan ekstraksi informasi canggih. Model seperti Gemini 2.5, GPT‑4V, Sonnet 3.7, dan Llama 4 merupakan contoh kemampuan ini. Misalnya, model multimodal dapat menganalisis gambar bagan bersama dengan pertanyaan teks untuk memberikan wawasan tentang tren data, menggabungkan pemahaman visual dan tekstual dalam aliran pemrosesan tunggal.

Kemampuan generasi konten, sebaliknya, berfokus pada pembuatan jenis media tertentu, seringkali dengan kualitas luar biasa tetapi fungsionalitas yang lebih khusus. Model teks‑ke‑gambar membuat konten visual dari deskripsi, sistem teks‑ke‑video menghasilkan klip video dari prompt, alat teks‑ke‑audio menghasilkan musik atau ucapan, dan model gambar‑ke‑gambar mengubah visual yang ada. Contoh termasuk Midjourney, DALL‑E, dan Stable Diffusion untuk gambar; Sora dan Pika untuk video; dan Suno dan ElevenLabs untuk audio. Tidak seperti model multimodal sejati, banyak sistem generasi berspesialisasi untuk modalitas output spesifik mereka, bahkan jika mereka dapat menerima beberapa jenis input. Mereka unggul dalam penciptaan alih‑alih pemahaman.

Karena LLM berevolusi melampaui teks, LangChain berkembang untuk mendukung alur kerja pemahaman dan generasi konten multimodal. Kerangka kerja ini memberi pengembang alat untuk menggabungkan kemampuan canggih ini ke dalam aplikasi mereka tanpa perlu mengimplementasikan integrasi kompleks dari awal. Mari kita mulai dengan menghasilkan gambar dari deskripsi teks. LangChain menyediakan beberapa pendekatan untuk menggabungkan generasi gambar melalui integrasi dan pembungkus eksternal. Kami akan menjelajahi beberapa pola implementasi, mulai dari yang paling sederhana dan berkembang ke teknik yang lebih canggih yang dapat dimasukkan ke dalam aplikasi Anda.

## Teks‑ke‑gambar

LangChain terintegrasi dengan berbagai model dan layanan generasi gambar, memungkinkan Anda:

- Menghasilkan gambar dari deskripsi teks
- Mengedit gambar yang ada berdasarkan prompt teks
- Mengontrol parameter generasi gambar
- Menangani variasi dan gaya gambar

LangChain menyertakan pembungkus dan model untuk layanan generasi gambar populer. Pertama, mari kita lihat cara menghasilkan gambar dengan seri model DALL‑E OpenAI.

### Menggunakan DALL‑E melalui OpenAI

Pembungkus LangChain untuk DALL‑E menyederhanakan proses menghasilkan gambar dari prompt teks. Implementasi menggunakan API OpenAI di balik layar tetapi memberikan antarmuka terstandarisasi yang konsisten dengan komponen LangChain lainnya.

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
dalle = DallEAPIWrapper(
   model_name="dall‑e‑3",  # Opsi: "dall‑e‑2" (default) atau "dall‑e‑3"
   size="1024x1024",       # Dimensi gambar
    quality="standard",     # "standard" atau "hd" untuk DALL‑E 3
    n=1                     # Jumlah gambar yang akan dihasilkan (hanya untuk DALL‑E 2)
)
# Hasilkan gambar
image_url = dalle.run("Diagram teknis terperinci komputer kuantum")
# Tampilkan gambar di notebook
from IPython.display import Image, display
display(Image(url=image_url))
# Atau simpan secara lokal
import requests
response = requests.get(image_url)
with open("generated_library.png", "wb") as f:
    f.write(response.content)
```

Inilah gambar yang kami dapatkan:

![Gambar 2.2: Gambar yang dihasilkan oleh OpenAI's DALL‑E Image Generator](Images/B32363_02_02.png)

Anda mungkin memperhatikan bahwa generasi teks dalam gambar‑gambar ini bukan salah satu keunggulan model ini. Anda dapat menemukan banyak model untuk generasi gambar di Replicate, termasuk model Stable Diffusion terkini, jadi ini yang akan kita gunakan sekarang.

### Menggunakan Stable Diffusion

Stable Diffusion 3.5 Large adalah model teks‑ke‑gambar terbaru Stability AI, dirilis pada Maret 2024. Ini adalah **Multimodal Diffusion Transformer** (**MMDiT**) yang menghasilkan gambar resolusi tinggi dengan detail dan kualitas luar biasa.

Model ini menggunakan tiga pengkode teks yang telah dilatih sebelumnya tetap dan mengimplementasikan Query‑Key Normalization untuk stabilitas pelatihan yang ditingkatkan. Ia mampu menghasilkan output beragam dari prompt yang sama dan mendukung berbagai gaya artistik.

```python
from langchain_community.llms import Replicate
# Inisialisasi model teks‑ke‑gambar dengan Stable Diffusion 3.5 Large
text2image = Replicate(
    model="stability‑ai/stable‑diffusion‑3.5‑large",
    model_kwargs={
        "prompt_strength": 0.85,
        "cfg": 4.5,
        "steps": 40,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "output_quality": 90
    }
)
# Hasilkan gambar
image_url = text2image.invoke(
    "Diagram teknis terperinci agen AI"
)
```

Parameter yang direkomendasikan untuk model baru termasuk:

- **prompt_strength**: Mengontrol seberapa dekat gambar mengikuti prompt (0.85)
- **cfg**: Mengontrol seberapa ketat model mengikuti prompt (4.5)
- **steps**: Lebih banyak langkah menghasilkan gambar berkualitas lebih tinggi (40)
- **aspect_ratio**: Diatur ke 1:1 untuk gambar persegi
- **output_format**: Menggunakan WebP untuk rasio kualitas‑ke‑ukuran yang lebih baik
- **output_quality**: Diatur ke 90 untuk output berkualitas tinggi

Inilah gambar yang kami dapatkan:

![Gambar 2.3: Gambar yang dihasilkan oleh Stable Diffusion](Images/B32363_02_03.png)

Sekarang mari kita jelajahi cara menganalisis dan memahami gambar menggunakan model multimodal.

## Pemahaman gambar

Pemahaman gambar mengacu pada kemampuan sistem AI untuk menafsirkan dan menganalisis informasi visual dengan cara yang mirip dengan persepsi visual manusia. Tidak seperti visi komputer tradisional (yang berfokus pada tugas spesifik seperti deteksi objek atau pengenalan wajah), model multimodal modern dapat melakukan penalaran umum tentang gambar, memahami konteks, hubungan, dan bahkan makna implisit dalam konten visual.

Gemini 2.5 Pro dan GPT‑4 Vision, di antara model lainnya, dapat menganalisis gambar dan memberikan deskripsi terperinci atau menjawab pertanyaan tentang mereka.

### Menggunakan Gemini 1.5 Pro

LangChain menangani input multimodal melalui antarmuka `ChatModel` yang sama. Ia menerima `Messages` sebagai input, dan objek `Message` memiliki bidang `content`. `content` dapat terdiri dari beberapa bagian, dan setiap bagian dapat mewakili modalitas berbeda (yang memungkinkan Anda mencampur modalitas berbeda dalam prompt Anda).

Anda dapat mengirim input multimodal berdasarkan nilai atau referensi. Untuk mengirimnya berdasarkan nilai, Anda harus mengencode byte sebagai string dan membuat variabel `image_url` yang diformat seperti dalam contoh di bawah menggunakan gambar yang kami hasilkan menggunakan Stable Diffusion:

```python
import base64
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages.human import HumanMessage
with open("stable‑diffusion.png", 'rb') as image_file:
    image_bytes = image_file.read()
    base64_bytes = base64.b64encode(image_bytes).decode("utf‑8")
prompt = [
   {"type": "text", "text": "Deskripsikan gambar: "},
   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_bytes}"}},
]
llm = ChatGoogleGenerativeAI(
    model="gemini‑1.5‑pro",
    temperature=0,
)
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
```

```
Gambar ini menampilkan penggambaran futuristik dan distilisasi tubuh atas robot humanoid dengan latar belakang tampilan digital biru bercahaya. Kepala robot bulat dan didominasi putih, dengan bagian material gelap, mungkin logam, di sekitar wajah dan telinga.  Wajah itu sendiri memiliki mata oranye bercahaya dan desain minimalis halus, tanpa hidung atau mulut dalam arti manusia tradisional.  Titik‑titik kecil terang, mungkin LED atau sensor, tersebar di kepala dan tubuh, menyarankan teknologi canggih dan konstruksi rumit.
Leher dan bahu robot terlihat, mengungkapkan struktur internal kompleks dari bagian‑bagian gelap yang saling terhubung, mungkin kabel atau kabel, yang kontras dengan eksterior putih. Bahu dan dada atas juga putih, dengan titik‑titik bercahaya serupa dan petunjuk mekanisme internal yang terlihat. Kesan keseluruhan adalah mesin yang ramping dan canggih.
Latar belakang adalah kisi berbagai antarmuka digital, menampilkan grafik, bagan, dan visualisasi data abstrak lainnya. Elemen‑elemen ini semua dalam nuansa biru, menciptakan suasana teknologi yang sejuk yang melengkapi penampilan robot. Tampilan bervariasi dalam ukuran dan kompleksitas, menambah kesan panel kontrol atau sistem pemantauan yang canggih. Kombinasi robot dan latar belakang menyarankan tema robotika canggih, kecerdasan buatan, atau analisis data.
```

Karena input multimodal biasanya memiliki ukuran besar, mengirim byte mentah sebagai bagian dari permintaan Anda mungkin bukan ide terbaik. Anda dapat mengirimnya berdasarkan referensi dengan menunjuk ke penyimpanan blob, tetapi jenis penyimpanan spesifik tergantung pada penyedia model. Misalnya, Gemini menerima input multimedia sebagai referensi ke Google Cloud Storage – layanan penyimpanan blob yang disediakan oleh Google Cloud.

```python
prompt = [
   {"type": "text", "text": "Deskripsikan video dalam beberapa kalimat."},
   {"type": "media", "file_uri": video_uri, "mime_type": "video/mp4"},
]
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
```

Detail pasti tentang cara membuat input multimodal mungkin tergantung pada penyedia LLM (dan integrasi LangChain yang sesuai menangani kamus yang sesuai dengan bagian bidang `content`). Misalnya, Gemini menerima kunci tambahan `"video_metadata"` yang dapat menunjuk ke offset awal dan/atau akhir potongan video yang akan dianalisis:

```python
offset_hint = {
           "start_offset": {"seconds": 10},
           "end_offset": {"seconds": 20},
       }
prompt = [
   {"type": "text", "text": "Deskripsikan video dalam beberapa kalimat."},
   {"type": "media", "file_uri": video_uri, "mime_type": "video/mp4", "video_metadata": offset_hint},
]
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
```

Dan, tentu saja, bagian multimodal seperti itu juga dapat ditemplatkan. Mari kita tunjukkan dengan templat sederhana yang mengharapkan argumen `image_bytes_str` yang berisi byte yang diencode:

```python
prompt = ChatPromptTemplate.from_messages(
   [("user",
    [{"type": "image_url",
      "image_url": {"url": "data:image/jpeg;base64,{image_bytes_str}"},
      }])]
)
prompt.invoke({"image_bytes_str": "test‑url"})
```

### Menggunakan GPT‑4 Vision

Setelah menjelajahi generasi gambar, mari kita periksa bagaimana LangChain menangani pemahaman gambar menggunakan model multimodal. Kemampuan GPT‑4 Vision (tersedia dalam model seperti GPT‑4o dan GPT‑4o‑mini) memungkinkan kita menganalisis gambar bersama teks, mengaktifkan aplikasi yang dapat "melihat" dan bernalar tentang konten visual.

LangChain menyederhanakan bekerja dengan model ini dengan menyediakan antarmuka konsisten untuk input multimodal. Mari kita implementasikan penganalisis gambar yang fleksibel:

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
def analyze_image(image_url: str, question: str) -> str:
    chat = ChatOpenAI(model="gpt‑4o‑mini", max_tokens=256)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "auto"
                }
            }
        ]
    )

    response = chat.invoke([message])
    return response.content
# Contoh penggunaan
image_url = "https://github.com/benman1/generative_ai_with_langchain/blob/f8f1680a8e5abf340dec4d02e38f7c3f84f02b41/chapter2/skyscrapers.png"
questions = [
    "Objek apa yang Anda lihat dalam gambar ini?",
    "Apa suasana atau atmosfer keseluruhan?",
    "Apakah ada orang dalam gambar?"
]
for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {analyze_image(image_url, question)}")
```

Model memberikan analisis kaya dan terperinci tentang pemandangan kota yang kami hasilkan:

```
Q: Objek apa yang Anda lihat dalam gambar ini?
A: Gambar ini menampilkan pemandangan kota futuristik dengan gedung pencakar langit tinggi dan ramping. Bangunan‑bangunan tampak memiliki efek bercahaya atau neon, menyarankan lingkungan berteknologi tinggi. Ada matahari atau sumber cahaya besar terang di langit, menambah suasana bersemangat. Jalan atau jalur terlihat di latar depan, mengarah ke kota, mungkin dengan garis‑garis cahaya yang menunjukkan gerakan atau kecepatan. Secara keseluruhan, adegan ini menyampaikan lanskap perkotaan lain dunia yang dinamis.
Q: Apa suasana atau atmosfer keseluruhan?
A: Suasana atau atmosfer keseluruhan dari adegan ini adalah futuristik dan bersemangat. Garis luar bercahaya gedung pencakar langit dan matahari terbenam terang menciptakan rasa energi dan kemungkinan. Kombinasi warna dalam dan cahaya menambah nada dramatis namun penuh harapan, menyarankan lingkungan perkotaan yang dinamis dan berkembang.
Q: Apakah ada orang dalam gambar?
A: Tidak ada orang dalam gambar. Tampaknya adalah pemandangan kota futuristik dengan bangunan tinggi dan matahari terbenam.
```

Kemampuan ini membuka banyak kemungkinan untuk aplikasi LangChain. Dengan menggabungkan analisis gambar dengan pola pemrosesan teks yang kita jelajahi sebelumnya dalam bab ini, Anda dapat membangun aplikasi canggih yang bernalar di seluruh modalitas. Di bab berikutnya, kita akan membangun konsep‑konsep ini untuk membuat aplikasi multimodal yang lebih canggih.

## Ringkasan

Setelah menyiapkan lingkungan pengembangan dan mengonfigurasi kunci API yang diperlukan, kita telah menjelajahi fondasi pengembangan LangChain, dari rantai dasar hingga kemampuan multimodal. Kita telah melihat bagaimana LCEL menyederhanakan alur kerja kompleks dan bagaimana LangChain terintegrasi dengan pemrosesan teks dan gambar. Blok bangunan ini mempersiapkan kita untuk aplikasi yang lebih canggih di bab‑bab mendatang.

Di bab berikutnya, kita akan memperluas konsep‑konsep ini untuk membuat aplikasi multimodal yang lebih canggih dengan alur kontrol yang ditingkatkan, output terstruktur, dan teknik prompt lanjutan. Anda akan mempelajari cara menggabungkan banyak modalitas dalam rantai kompleks, memasukkan penanganan kesalahan yang lebih canggih, dan membangun aplikasi yang memanfaatkan potensi penuh LLM modern.

## Pertanyaan ulasan

1. Apa tiga keterbatasan utama LLM mentah yang diatasi LangChain?
   - Keterbatasan memori
   - Integrasi alat
   - Batasan konteks
   - Kecepatan pemrosesan
   - Optimasi biaya
2. Manakah dari berikut ini yang paling baik menggambarkan tujuan LCEL (Bahasa Ekspresi LangChain)?
   - Bahasa pemrograman untuk LLM
   - Antarmuka terpadu untuk menyusun komponen LangChain
   - Sistem template untuk prompt
   - Kerangka kerja pengujian untuk LLM
3. Sebutkan tiga jenis sistem memori yang tersedia di LangChain
4. Bandingkan dan bedakan LLM dan model obrolan di LangChain. Bagaimana perbedaan antarmuka dan kasus penggunaannya?
5. Peran apa yang dimainkan Runnables di LangChain? Bagaimana kontribusinya dalam membangun aplikasi LLM modular?
6. Saat menjalankan model lokal, faktor mana yang memengaruhi kinerja model? (Pilih semua yang berlaku)
   - RAM yang tersedia
   - Kemampuan CPU/GPU
   - Kecepatan koneksi internet
   - Tingkat kuantisasi model
   - Jenis sistem operasi
7. Bandingkan opsi penyebaran model berikut dan identifikasi skenario di mana masing‑masing paling tepat:
   - Model berbasis cloud (mis., OpenAI)
   - Model lokal dengan llama.cpp
   - Integrasi GPT4All
8. Rancang rantai dasar menggunakan LCEL yang akan:
   - Mengambil pertanyaan pengguna tentang produk
   - Mengueri database untuk informasi produk
   - Menghasilkan respons menggunakan LLM
9. Berikan sketsa yang menguraikan komponen dan cara penghubungannya.
10. Bandingkan pendekatan berikut untuk analisis gambar dan sebutkan pertukaran di antara mereka:
    - Pendekatan A
      ```python
      from langchain_openai import ChatOpenAI
      chat = ChatOpenAI(model="gpt‑4‑vision‑preview")
      ```
    - Pendekatan B
      ```python
      from langchain_community.llms import Ollama
      local_model = Ollama(model="llava")
      ```

## Berlangganan newsletter mingguan kami

Berlangganan AI_Distilled, newsletter untuk profesional AI, peneliti, dan inovator, di [https://packt.link/Q5UyU](https://packt.link/Q5UyU).

![Kode QR](Images/Newsletter_QRcode1.jpg)

[^70]: Penanda indeks untuk dependensi
[^71]: Penanda indeks untuk contoh Colab
[^72]: Penanda indeks untuk notebook Kaggle
[^73]: Penanda indeks untuk notebook Gradient
[^74]: Penanda indeks untuk Miniconda
[^75]: Penanda indeks untuk pilihan metode
[^76]: Penanda indeks untuk Docker
[^77]: Penanda indeks untuk Conda
[^78]: Penanda indeks untuk Poetry
[^79]: Penanda indeks untuk LangChain
[^80]: Penanda indeks untuk penyedia
[^81]: Penanda indeks untuk penyedia
[^82]: Penanda indeks untuk kunci lingkungan
[^83]: Penanda indeks untuk file konfigurasi
[^84]: Penanda indeks untuk model OpenAI
[^85]: Penanda indeks untuk LLM
[^86]: Penanda indeks untuk Claude
[^87]: Penanda indeks untuk model Mistral
[^88]: Penanda indeks untuk Gemini
[^89]: Penanda indeks untuk OpenAI GPT
[^90]: Penanda indeks untuk model DeepSeek
[^91]: Penanda indeks untuk AI21 Labs
[^92]: Penanda indeks untuk Inflection Pi
[^93]: Penanda indeks untuk model Perplexity
[^94]: Penanda indeks untuk model Cohere
[^95]: Penanda indeks untuk Bedrock
[^96]: Penanda indeks untuk lainnya
[^97]: Penanda indeks untuk Azure OpenAI
[^98]: Penanda indeks untuk Vertex AI
[^99]: Penanda indeks untuk model sumber terbuka
[^100]: Penanda indeks untuk Replicate
[^101]: Penanda indeks untuk HuggingFace
[^102]: Penanda indeks untuk fleksibilitas
[^103]: Penanda indeks untuk aplikasi praktis
[^104]: Penanda indeks untuk antarmuka LLM
[^105]: Penanda indeks untuk implementasi
[^106]: Penanda indeks untuk antarmuka LLM
[^107]: Penanda indeks untuk Gemini
[^108]: Penanda indeks untuk pengembangan
[^109]: Penanda indeks untuk model obrolan
[^110]: Penanda indeks untuk tipe pesan
[^111]: Penanda indeks untuk penyelesaian loop
[^112]: Penanda indeks untuk Claude 3.7 Sonnet
[^113]: Penanda indeks untuk respons
[^114]: Penanda indeks untuk prompt penalaran
[^115]: Penanda indeks untuk kontrol
[^116]: Penanda indeks untuk parameter
[^117]: Penanda indeks untuk perusahaan
[^118]: Penanda indeks untuk rekayasa prompt
[^119]: Penanda indeks untuk prompt
[^120]: Penanda indeks untuk template prompt
[^121]: Penanda indeks untuk template
[^122]: Penanda indeks untuk model obrolan
[^123]: Penanda indeks untuk LCEL
[^124]: Penanda indeks untuk evolusi
[^125]: Penanda indeks untuk LangChain
[^126]: Penanda indeks untuk transformasi
[^127]: Penanda indeks untuk alur kerja canggih
[^128]: Penanda indeks untuk eksplorasi
[^129]: Penanda indeks untuk dapat disusun
[^130]: Penanda indeks untuk tantangan
[^131]: Penanda indeks untuk LCEL
[^132]: Penanda indeks untuk konstruksi
[^133]: Penanda indeks untuk sederhana
[^134]: Penanda indeks untuk berhasil
[^135]: Penanda indeks untuk contoh
[^136]: Penanda indeks untuk aplikasi
[^137]: Penanda indeks untuk model lokal
[^138]: Penanda indeks untuk Ollama
[^139]: Penanda indeks untuk menyediakan
[^140]: Penanda indeks untuk LCEL
[^141]: Penanda indeks untuk fungsi
[^142]: Penanda indeks untuk Hugging Face
[^143]: Penanda indeks untuk Hugging Face Hub
[^144]: Penanda indeks untuk dihilangkan
[^145]: Penanda indeks untuk LangChain
[^146]: Penanda indeks untuk llama.cpp
[^147]: Penanda indeks untuk GPT4All
[^148]: Penanda indeks untuk bekerja
[^149]: Penanda indeks untuk model lokal
[^150]: Penanda indeks untuk model lokal
[^151]: Penanda indeks untuk LangChain
[^152]: Penanda indeks untuk sistem AI
[^153]: Penanda indeks untuk LangChain
[^154]: Penanda indeks untuk model generasi gambar
[^155]: Penanda indeks untuk pembungkus
[^156]: Penanda indeks untuk DALL‑E
[^157]: Penanda indeks untuk menyederhanakan
[^158]: Penanda indeks untuk memperhatikan
[^159]: Penanda indeks untuk gambar‑gambar ini
[^160]: Penanda indeks untuk Stable Diffusion
[^161]: Penanda indeks untuk terbaru
[^162]: Penanda indeks untuk menghasilkan
[^163]: Penanda indeks untuk direkomendasikan
[^164]: Penanda indeks untuk model
[^165]: Penanda indeks untuk Gambar
[^166]: Penanda indeks untuk kemampuan
[^167]: Penanda indeks untuk LangChain
[^168]: Penanda indeks untuk sama
[^169]: Penanda indeks untuk kirim
[^170]: Penanda indeks untuk string
[^171]: Penanda indeks untuk input multimodal
[^172]: Penanda indeks untuk menunjuk
[^173]: Penanda indeks untuk setelah
[^174]: Penanda indeks untuk LangChain
[^175]: Penanda indeks untuk model
[^176]: Penanda indeks untuk kemampuan
