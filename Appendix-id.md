# Lampiran

Lampiran ini berfungsi sebagai panduan referensi praktis untuk penyedia LLM besar yang terintegrasi dengan LangChain. Saat kamu mengembangkan aplikasi dengan teknik yang dibahas di sepanjang buku ini, kamu perlu terhubung ke berbagai penyedia model, masing‑masing dengan mekanisme otentikasi, kemampuan, dan pola integrasinya sendiri‑sendiri.

Kita akan pertama‑tama membahas petunjuk penyiapan rinci untuk penyedia LLM utama, termasuk OpenAI, Hugging Face, Google, dan lainnya. Untuk setiap penyedia, kita akan menjelajahi proses pembuatan akun, pembuatan kunci API, dan pengonfigurasian lingkungan pengembanganmu untuk menggunakan layanan tersebut dengan LangChain. Kemudian kita simpulkan dengan contoh implementasi praktis yang menunjukkan cara memproses konten yang melampaui jendela konteks LLM—khususnya, meringkas video panjang menggunakan teknik peta‑reduksi dengan LangChain. Pola ini dapat disesuaikan untuk berbagai skenario di mana kamu perlu memproses volume teks, transkrip audio, atau konten lain yang tidak muat dalam satu konteks LLM.

## OpenAI

OpenAI tetap menjadi salah satu penyedia LLM paling populer, menawarkan model dengan berbagai tingkat kekuatan yang cocok untuk tugas‑tugas berbeda, termasuk GPT‑4 dan GPT‑o1. LangChain menyediakan integrasi mulus dengan API OpenAI, mendukung baik model penyelesaian tradisional maupun model obrolan mereka. Setiap model ini memiliki harganya sendiri, biasanya per token.

Untuk bekerja dengan model OpenAI, kita perlu mendapatkan kunci API OpenAI terlebih dahulu. Untuk membuat kunci API, ikuti langkah‑langkah ini:

1. Kamu perlu membuat login di [https://platform.openai.com/](https://platform.openai.com/).
2. Atur informasi penagihanmu.
3. Kamu dapat melihat kunci API di bawah **Personal** | **View API Keys**.
4. Klik **Create new secret key** dan beri nama.

Begini seharusnya tampilannya di platform OpenAI:

![Gambar A.1: Platform API OpenAI – Buat kunci rahasia baru](Images/B32363_Appendix_01-01.png)

Gambar A.1: Platform API OpenAI – Buat kunci rahasia baru

Setelah mengklik **Create secret key**, kamu akan melihat pesan “API key generated”. Kamu perlu menyalin kunci ke papan klip dan menyimpannya, karena kamu akan membutuhkannya. Kamu dapat menetapkan kunci sebagai variabel lingkungan (**OPENAI_API_KEY**) atau meneruskannya sebagai parameter setiap kali kamu membuat kelas untuk panggilan OpenAI.

Kamu dapat menentukan model yang berbeda saat menginisialisasi modelmu, baik itu model obrolan maupun LLM. Kamu dapat melihat daftar model di [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models).

OpenAI menyediakan rangkaian kemampuan komprehensif yang terintegrasi mulus dengan LangChain, termasuk:

- Model bahasa inti melalui API OpenAI
- Kelas penyematan untuk model penyematan teks

Kita akan membahas dasar‑dasar integrasi model di bab ini, sementara eksplorasi mendalam tentang fitur khusus seperti penyematan, asisten, dan moderasi akan mengikuti di Bab 4 dan 5.

## Hugging Face

Hugging Face adalah pemain yang sangat menonjol di bidang NLP dan memiliki daya tarik yang cukup besar dalam solusi sumber terbuka dan hosting. Perusahaan ini adalah perusahaan Prancis‑Amerika yang mengembangkan alat untuk membangun aplikasi ML. Para karyawannya mengembangkan dan memelihara pustaka Python Transformers, yang digunakan untuk tugas‑tugas NLP, mencakup implementasi model mutakhir dan populer seperti Mistral 7B, BERT, dan GPT‑2, serta kompatibel dengan PyTorch, TensorFlow, dan JAX.

Di samping produk mereka, Hugging Face terlibat dalam inisiatif seperti Lokakarya Penelitian BigScience, di mana mereka merilis LLM terbuka bernama BLOOM dengan 176 miliar parameter. Hugging Face juga telah menjalin kemitraan dengan perusahaan seperti Graphcore dan Amazon Web Services untuk mengoptimalkan penawaran mereka dan membuatnya tersedia untuk basis pelanggan yang lebih luas.

LangChain mendukung pemanfaatan Hugging Face Hub, yang memberikan akses ke sejumlah besar model, kumpulan data dalam berbagai bahasa dan format, serta aplikasi demo. Ini termasuk integrasi dengan Hugging Face Endpoints, yang memungkinkan inferensi pembuatan teks yang didukung oleh layanan Text Generation Inference. Pengguna dapat terhubung ke berbagai jenis Endpoint, termasuk API Serverless Endpoints gratis dan Inference Endpoints khusus untuk beban kerja perusahaan yang dilengkapi dukungan AutoScaling.

Untuk penggunaan lokal, LangChain menyediakan integrasi dengan model dan pipa Hugging Face. Kelas `ChatHuggingFace` memungkinkan penggunaan model Hugging Face untuk aplikasi obrolan, sementara kelas `HuggingFacePipeline` memungkinkan menjalankan model Hugging Face secara lokal melalui pipa. Selain itu, LangChain mendukung model penyematan dari Hugging Face, termasuk `HuggingFaceEmbeddings`, `HuggingFaceInstructEmbeddings`, dan `HuggingFaceBgeEmbeddings`.

Kelas `HuggingFaceHubEmbeddings` memungkinkan pemanfaatan toolkit **Text Embeddings Inference** (**TEI**) Hugging Face untuk ekstraksi berkinerja tinggi. LangChain juga menyediakan `HuggingFaceDatasetLoader` untuk memuat kumpulan data dari Hugging Face Hub.

Untuk menggunakan Hugging Face sebagai penyedia modelmu, kamu dapat membuat akun dan kunci API di [https://huggingface.co/settings/profile](https://huggingface.co/settings/profile). Selain itu, kamu dapat membuat token tersedia di lingkunganmu sebagai `HUGGINGFACEHUB_API_TOKEN`.

## Google

Google menawarkan dua platform utama untuk mengakses LLM‑nya, termasuk model Gemini terbaru:

### 1. Platform Google AI

Platform Google AI menyediakan penyiapan yang sederhana bagi pengembang dan pengguna, serta akses ke model Gemini terbaru. Untuk menggunakan model Gemini melalui Google AI:

- **Akun Google**: Akun Google standar sudah cukup untuk otentikasi.
- **Kunci API**: Hasilkan kunci API untuk mengotentikasi permintaanmu.
  - Kunjungi halaman ini untuk membuat kunci API‑mu: [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key)
  - Setelah mendapatkan kunci API, atur variabel lingkungan `GOOGLE_API_KEY` di lingkungan pengembanganmu (lihat petunjuk untuk OpenAI) untuk mengotentikasi permintaanmu.

### 2. Google Cloud Vertex AI

Untuk fitur dan integrasi tingkat perusahaan, model Gemini Google tersedia melalui platform Vertex AI Google Cloud. Untuk menggunakan model melalui Vertex AI:

1. Buat akun Google Cloud, yang memerlukan penerimaan persyaratan layanan dan penyiapan penagihan.
2. Instal CLI gcloud untuk berinteraksi dengan layanan Google Cloud. Ikuti petunjuk instalasi di [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install).
3. Jalankan perintah berikut untuk mengotentikasi dan mendapatkan token kunci:
   ```
   gcloud auth application-default login
   ```
4. Pastikan API Vertex AI diaktifkan untuk proyek Google Cloud‑mu.
5. Kamu dapat menetapkan ID proyek Google Cloud‑mu – misalnya, menggunakan perintah `gcloud`:
   ```
   gcloud config set project my-project
   ```

Metode lain adalah dengan meneruskan argumen konstruktor saat menginisialisasi LLM, menggunakan aiplatform.init(), atau menetapkan variabel lingkungan GCP.

Kamu dapat membaca lebih lanjut tentang opsi‑opsi ini di dokumentasi Vertex.

Jika kamu belum mengaktifkan layanan yang relevan, kamu akan mendapatkan pesan kesalahan yang membantu yang mengarahkanmu ke situs web yang tepat, di mana kamu klik **Enable**. Kamu harus mengaktifkan Vertex atau API Generative Language sesuai preferensi dan ketersediaan.

LangChain menawarkan integrasi dengan layanan Google seperti inferensi model bahasa, penyematan, pengingesan data dari berbagai sumber, transformasi dokumen, dan terjemahan.

> Ada dua paket integrasi utama:
>
> - `langchain-google-vertexai`
> - `langchain-google-genai`
>
> Kita akan menggunakan `langchain-google-genai`, paket yang direkomendasikan oleh LangChain untuk pengembang individu. Penyiapannya sederhana, hanya memerlukan akun Google dan kunci API. Disarankan untuk beralih ke `langchain-google-vertexai` untuk proyek yang lebih besar. Integrasi ini menawarkan fitur perusahaan seperti kunci enkripsi pelanggan, integrasi virtual private cloud, dan lainnya, yang memerlukan akun Google Cloud dengan penagihan.
>
> Jika kamu telah mengikuti petunjuk di GitHub, seperti yang ditunjukkan di bagian sebelumnya, kamu seharusnya sudah memasang paket `langchain-google-genai`.

## Penyedia lainnya

- **Replicate**: Kamu dapat mengotentikasi dengan kredensial GitHub‑mu di [https://replicate.com/](https://replicate.com/). Jika kemudian kamu mengklik ikon pengguna di kiri atas, kamu akan menemukan token API – cukup salin kunci API dan buat tersedia di lingkunganmu sebagai `REPLICATE_API_TOKEN`. Untuk menjalankan pekerjaan yang lebih besar, kamu perlu menyiapkan kartu kredit‑mu (di bawah penagihan).
- **Azure**: Dengan mengotentikasi baik melalui GitHub atau kredensial Microsoft, kita dapat membuat akun di Azure di [https://azure.microsoft.com/](https://azure.microsoft.com/). Kita kemudian dapat membuat kunci API baru di bawah **Cognitive Services** | **Azure OpenAI**.
- **Anthropic**: Kamu perlu menetapkan variabel lingkungan `ANTHROPIC_API_KEY`. Pastikan kamu telah menyiapkan penagihan dan menambahkan dana di konsol Anthropic di [https://console.anthropic.com/](https://console.anthropic.com/).

## Meringkas video panjang

Di [Bab 3](Chapter_3.xhtml#_idTextAnchor049), kita menunjukkan cara meringkas video panjang (yang tidak muat dalam jendela konteks) dengan pendekatan peta‑reduksi. Kita menggunakan LangGraph untuk merancang alur kerja seperti itu. Tentu saja, kamu dapat menggunakan pendekatan yang sama untuk kasus serupa – misalnya, untuk meringkas teks panjang atau untuk mengekstrak informasi dari audio panjang. Sekarang mari kita lakukan hal yang sama hanya dengan LangChain, karena ini akan menjadi latihan berguna yang akan membantu kita lebih memahami beberapa internal kerangka kerja.

Pertama, sebuah `PromptTemplate` tidak mendukung jenis media (per Februari 2025), jadi kita perlu mengonversi input menjadi daftar pesan secara manual. Untuk menggunakan rantai berparameter, sebagai jalan keluar, kita akan membuat fungsi Python yang mengambil argumen (selalu diberikan berdasarkan nama) dan membuat daftar pesan untuk diproses. Setiap pesan menginstruksikan LLM untuk meringkas bagian tertentu dari video (dengan membaginya menjadi interval offset), dan pesan‑pesan ini dapat diproses secara paralel. Keluaran akan berupa daftar string, masing‑ masing meringkas bagian kecil dari video asli.

Saat kamu menggunakan tanda bintang tambahan (\*) dalam deklarasi fungsi Python, itu berarti argumen setelah tanda bintang harus diberikan hanya berdasarkan nama. Misalnya, mari buat fungsi sederhana dengan banyak argumen yang dapat kita panggil dengan berbagai cara di Python dengan hanya meneruskan beberapa (atau tidak ada) parameter berdasarkan nama:

```python
def test(a: int, b: int = 2, c: int = 3):
    print(f"a={a}, b={b}, c={c}")
    pass
test(1, 2, 3)
test(1, 2, c=3)
test(1, b=2, c=3)
test(1, c=3)
```

Tetapi jika kamu mengubah tanda tangannya, pemanggilan pertama akan menimbulkan kesalahan:

```python
def test(a: int, b: int = 2, *, c: int = 3):
    print(f"a={a}, b={b}, c={c}")
    pass
# ini tidak berfungsi lagi: test(1, 2, 3)
```

Kamu mungkin sering melihat ini jika melihat kode sumber LangChain. Itulah mengapa kami memutuskan untuk menjelaskannya sedikit lebih rinci.

Sekarang, kembali ke kode kita. Kita masih perlu menjalankan dua langkah terpisah jika kita ingin meneruskan `video_uri` sebagai argumen input. Tentu saja, kita dapat membungkus langkah‑langkah ini sebagai fungsi Python, tetapi sebagai alternatif, kita gabungkan semuanya menjadi satu rantai:

```python
from langchain_core.runnables import RunnableLambda
create_inputs_chain = RunnableLambda(lambda x: _create_input_
messages(**x))
map_step_chain = create_inputs_chain | RunnableLambda(lambda x: map_chain.
batch(x, config={"max_concurrency": 3}))
summaries = map_step_chain.invoke({"video_uri": video_uri})
```

Sekarang mari gabungkan semua ringkasan yang diberikan menjadi satu prompt dan minta LLM untuk menyiapkan ringkasan akhir:

```python
def _merge_summaries(summaries: list[str], interval_secs: int = 600, **kwargs) -> str:
    sub_summaries = []
    for i, summary in enumerate(summaries):
        sub_summary = (
            f"Ringkasan dari detik {i*interval_secs} hingga detik {(i+1)*interval_secs}:"
            f"\n{summary}\n"
        )
        sub_summaries.append(sub_summary)
    return "".join(sub_summaries)
reduce_prompt = PromptTemplate.from_template(
    "Kamu diberi daftar ringkasan yang"
    "dari sebuah video yang dibagi menjadi potongan‑potongan berurutan.\n"
    "RINGKASAN:\n{summaries}"
    "Berdasarkan itu, siapkan ringkasan dari seluruh video."
)
reduce_chain = RunnableLambda(lambda x: _merge_summaries(**x)) | reduce_prompt | llm | StrOutputParser()
final_summary = reduce_chain.invoke({"summaries": summaries})
```

Untuk menggabungkan semuanya, kita memerlukan rantai yang pertama‑tama mengeksekusi semua langkah MAP dan kemudian fase REDUCE:

```python
from langchain_core.runnables import RunnablePassthrough
final_chain = (
    RunnablePassthrough.assign(summaries=map_step_chain).assign(final_ summary=reduce_chain)
    | RunnableLambda(lambda x: x["final_summary"])
)
result = final_chain.invoke({
    "video_uri": video_uri,
    "interval_secs": 300,
    "chunks": 9
})
```

Mari kita ulangi apa yang kita lakukan. Kita menghasilkan banyak ringkasan dari bagian‑bagian video yang berbeda, lalu kita meneruskan ringkasan‑ringkasan ini ke LLM sebagai teks dan memberikan tugas untuk membuat ringkasan akhir. Kita menyiapkan ringkasan setiap bagian secara independen kemudian menggabungkannya, yang memungkinkan kita mengatasi batasan ukuran jendela konteks untuk video dan mengurangi latensi banyak karena paralelisasi. Alternatif lain adalah pendekatan yang disebut **refine**. Kita mulai dengan ringkasan kosong dan melakukan peringkasan langkah demi langkah – setiap kali, menyediakan LLM dengan potongan video baru dan ringkasan yang sebelumnya dihasilkan sebagai input. Kami mendorong pembaca untuk membangun ini sendiri karena ini akan menjadi perubahan yang relatif sederhana pada kode.
