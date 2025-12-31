# Membangun Alur Kerja dengan LangGraph

Sejauh ini, kita sudah belajar tentang LLM, LangChain sebagai kerangka kerja, dan cara menggunakan LLM dengan LangChain dalam mode biasa (hanya meminta menghasilkan teks berdasarkan petunjuk). Di bab ini, kita akan mulai dengan pengenalan singkat tentang LangGraph sebagai kerangka kerja dan cara mengembangkan alur kerja yang lebih kompleks dengan LangChain dan LangGraph dengan merantai beberapa langkah. Sebagai contoh, kita akan membahas penguraian keluaran LLM dan melihat pola penanganan kesalahan dengan LangChain dan LangGraph. Lalu, kita akan lanjutkan dengan cara yang lebih canggih untuk mengembangkan petunjuk dan menjelajah blok-blok bangunan yang ditawarkan LangGraph untuk teknik few-shot prompting dan lainnya.

Kita juga akan membahas bekerja dengan input multimodal, memanfaatkan konteks panjang, dan menyesuaikan beban kerja untuk mengatasi batasan terkait ukuran jendela konteks. Akhirnya, kita akan melihat mekanisme dasar pengelolaan memori dengan LangChain. Memahami teknik dasar dan kunci ini akan membantu kita membaca kode LangGraph, memahami tutorial dan contoh kode, serta mengembangkan alur kerja kompleks kita sendiri. Tentu saja, kita akan bahas apa itu alur kerja LangGraph dan akan terus membangun keterampilan itu di _Bab 5_ dan _Bab 6_.

Singkatnya, kita akan membahas topik utama berikut di bab ini:

- Dasar-dasar LangGraph
- Rekayasa petunjuk
- Bekerja dengan jendela konteks pendek
- Memahami mekanisme memori

> Seperti biasa, kamu bisa temukan semua contoh kode di repositori GitHub publik kami sebagai notebook Jupyter: [https://github.com/benman1/generative_ai_with_langchain/tree/second_edition/chapter3](https://github.com/benman1/generative_ai_with_langchain/tree/second_edition/chapter3).

## Dasar-dasar LangGraph

LangGraph adalah kerangka kerja yang dikembangkan oleh LangChain (sebagai perusahaan) yang membantu mengendalikan dan mengatur alur kerja. Mengapa kita butuh kerangka kerja orkestrasi lain? Mari kita tunda pertanyaan ini sampai [Bab 5](Chapter_5.xhtml#_idTextAnchor111), di mana kita akan menyentuh agen dan alur kerja agen, tapi untuk sekarang, sebutkan fleksibilitas LangGraph sebagai kerangka kerja orkestrasi dan ketangguhannya dalam menangani skenario kompleks.

Tidak seperti banyak kerangka kerja lain, LangGraph mengizinkan siklus (kebanyakan kerangka kerja orkestrasi lain hanya beroperasi dengan graf asiklik langsung), mendukung streaming langsung, dan memiliki banyak perulangan dan komponen pra-bangun yang didedikasikan untuk aplikasi AI generatif (misalnya, moderasi manusia). LangGraph juga memiliki API yang sangat kaya yang memungkinkan kamu memiliki kendali sangat granular atas alur eksekusi jika diperlukan. Ini tidak sepenuhnya dibahas dalam buku kami, tapi ingat bahwa kamu selalu dapat menggunakan API yang lebih rendah jika perlu.

> **Graf Asiklik Berarah (Directed Acyclic Graph, DAG)** adalah jenis graf khusus dalam teori graf dan ilmu komputer. Sisi-sisinya (koneksi antar node) memiliki arah, artinya koneksi dari node A ke node B berbeda dari koneksi dari node B ke node A. Ia tidak memiliki siklus. Dengan kata lain, tidak ada jalur yang dimulai di suatu node dan kembali ke node yang sama dengan mengikuti sisi berarah.
>
> DAG sering digunakan sebagai model alur kerja dalam teknik data, di mana node adalah tugas dan sisi adalah ketergantungan antara tugas-tugas ini. Misalnya, sisi dari node A ke node B berarti kita butuh keluaran dari node A untuk mengeksekusi node B.

Untuk sekarang, mari mulai dengan dasar-dasar. Jika kamu baru dengan kerangka kerja ini, kami juga sangat merekomendasikan kursus online gratis tentang LangGraph yang tersedia di [https://academy.langchain.com/](https://academy.langchain.com/) untuk memperdalam pemahamanmu.

### Pengelolaan keadaan

Pengelolaan keadaan sangat penting dalam aplikasi AI dunia nyata. Misalnya, dalam chatbot layanan pelanggan, keadaan mungkin melacak informasi seperti ID pelanggan, riwayat percakapan, dan masalah yang belum selesai. Pengelolaan keadaan LangGraph memungkinkan kamu mempertahankan konteks ini di seluruh alur kerja kompleks dari beberapa komponen AI.

LangGraph memungkinkan kamu mengembangkan dan mengeksekusi alur kerja kompleks yang disebut **graf**. Kita akan gunakan kata _graf_ dan _alur kerja_ secara bergantian di bab ini. Graf terdiri dari node dan sisi di antaranya. Node adalah komponen alur kerjamu, dan alur kerja memiliki _keadaan_. Apa itu? Pertama, keadaan membuat node-mu sadar akan konteks saat ini dengan melacak input pengguna dan komputasi sebelumnya. Kedua, keadaan memungkinkan kamu mempertahankan eksekusi alur kerja-mu di titik mana pun. Ketiga, keadaan membuat alur kerja-mu benar-benar interaktif karena sebuah node dapat mengubah perilaku alur kerja dengan memperbarui keadaan. Untuk kesederhanaan, anggap keadaan sebagai kamus Python. Node adalah fungsi Python yang beroperasi pada kamus ini. Mereka mengambil kamus sebagai input dan mengembalikan kamus lain yang berisi kunci dan nilai untuk diperbarui dalam keadaan alur kerja.

Mari pahami dengan contoh sederhana. Pertama, kita perlu mendefinisikan skema keadaan:

```python
from typing_extensions import TypedDict
class JobApplicationState(TypedDict):
   job_description: str
   is_suitable: bool
   application: str
```

`TypedDict` adalah konstruktor tipe Python yang memungkinkan mendefinisikan kamus dengan kumpulan kunci yang telah ditentukan dan setiap kunci dapat memiliki tipenya sendiri (berbeda dengan konstruksi `Dict[str, str]`).

> Skema keadaan LangGraph tidak harus didefinisikan sebagai `TypedDict`; kamu juga bisa gunakan kelas data atau model Pydantic.

Setelah kita mendefinisikan skema untuk keadaan, kita bisa mendefinisikan alur kerja pertama kita yang sederhana:

```python
from langgraph.graph import StateGraph, START, END, Graph
def analyze_job_description(state):
   print("...Menganalisis deskripsi pekerjaan yang diberikan ...")
   return {"is_suitable": len(state["job_description"]) > 100}
def generate_application(state):
   print("...menghasilkan lamaran...")
   return {"application": "some_fake_application"}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application", END)
graph = builder.compile()
```

Di sini, kita mendefinisikan dua fungsi Python yang merupakan komponen alur kerja kita. Lalu, kita mendefinisikan alur kerja kita dengan memberikan skema keadaan, menambahkan node dan sisi di antaranya. `add_node` adalah cara mudah untuk menambahkan komponen ke grafik-mu (dengan memberikan namanya dan fungsi Python yang sesuai), dan kamu dapat merujuk nama ini nanti saat kamu mendefinisikan sisi dengan `add_edge`. `START` dan `END` adalah node bawaan yang dipesan yang mendefinisikan awal dan akhir alur kerja.

Mari lihat alur kerja kita dengan menggunakan mekanisme visualisasi bawaan:

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

![Gambar 3.1: Visualisasi bawaan LangGraph dari alur kerja pertama kita](Images/B32363_03_01.png)

Fungsi kita mengakses keadaan hanya dengan membaca dari kamus yang secara otomatis diberikan LangGraph sebagai input. LangGraph mengisolasi pembaruan keadaan. Ketika node menerima keadaan, ia mendapat salinan tidak berubah, bukan referensi ke objek keadaan sebenarnya. Node harus mengembalikan kamus yang berisi kunci dan nilai spesifik yang ingin diperbarui. LangGraph kemudian menangani penggabungan pembaruan ini ke keadaan utama. Pola ini mencegah efek samping dan memastikan bahwa perubahan keadaan eksplisit dan dapat dilacak.

Satu-satunya cara bagi node untuk memodifikasi keadaan adalah dengan memberikan kamus keluaran dengan pasangan kunci-nilai untuk diperbarui, dan LangGraph akan menanganinya. Sebuah node harus memodifikasi setidaknya satu kunci dalam keadaan. Instance `graph` sendiri adalah `Runnable` (tepatnya, ia mewarisi dari `Runnable`) dan kita dapat mengeksekusinya. Kita harus memberikan kamus dengan keadaan awal, dan kita akan mendapat keadaan akhir sebagai keluaran:

```python
res = graph.invoke({"job_description":"fake_jd"})
```

```
print(res)
>>...Menganalisis deskripsi pekerjaan yang diberikan ...
...menghasilkan lamaran...
{'job_description': 'fake_jd', 'is_suitable': True, 'application': 'some_fake_application'}
```

Kami menggunakan graf yang sangat sederhana sebagai contoh. Dengan alur kerja nyata-mu, kamu dapat mendefinisikan langkah paralel (misalnya, kamu dapat dengan mudah menghubungkan satu node dengan banyak node) dan bahkan siklus. LangGraph mengeksekusi alur kerja dalam apa yang disebut _superstep_ yang dapat memanggil beberapa node secara bersamaan (lalu menggabungkan pembaruan keadaan dari node-node ini). Kamu dapat mengontrol kedalaman rekursi dan jumlah total superstep dalam graf, yang membantumu menghindari siklus berjalan selamanya, terutama karena keluaran LLM tidak deterministik.

> **Superstep** di LangGraph mewakili iterasi diskrit atas satu atau beberapa node, dan terinspirasi oleh Pregel, sistem yang dibangun Google untuk memproses graf besar dalam skala. Ia menangani eksekusi paralel node dan pembaruan yang dikirim ke keadaan pusat graf.

Dalam contoh kita, kita menggunakan sisi langsung dari satu node ke node lain. Itu membuat graf kita tidak berbeda dari rantai sekuensial yang bisa kita definisikan dengan LangChain. Salah satu fitur kunci LangGraph adalah kemampuan untuk membuat sisi bersyarat yang dapat mengarahkan alur eksekusi ke satu node atau node lain bergantung pada keadaan saat ini. Sisi bersyarat adalah fungsi Python yang mendapatkan keadaan saat ini sebagai input dan mengembalikan string dengan nama node yang akan dieksekusi.

Mari lihat contoh:

```python
from typing import Literal
builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
   if state.get("is_suitable"):
       return "generate_application"
   return END
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)
graph = builder.compile()
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

Kita telah mendefinisikan sisi `is_suitable_condition` yang mengambil keadaan dan mengembalikan string `END` atau `generate_application` dengan menganalisis keadaan saat ini. Kita menggunakan petunjuk tipe `Literal` karena digunakan oleh LangGraph untuk menentukan node tujuan mana yang akan dihubungkan dengan node sumber saat membuat sisi bersyarat. Jika kamu tidak menggunakan petunjuk tipe, kamu dapat memberikan daftar node tujuan langsung ke fungsi `add_conditional_edges`; jika tidak, LangGraph akan menghubungkan node sumber dengan semua node lain dalam graf (karena tidak menganalisis kode fungsi sisi itu sendiri saat membuat graf). Gambar berikut menunjukkan keluaran yang dihasilkan:

![Gambar 3.2: Alur kerja dengan sisi bersyarat (diwakili sebagai garis putus-putus)](Images/B32363_03_02.png)

Sisi bersyarat divisualisasikan dengan garis putus-putus, dan sekarang kita dapat lihat bahwa, bergantung pada keluaran langkah `analyze_job_description`, graf kita dapat melakukan tindakan berbeda.

### Reducer

Sejauh ini, node kita telah mengubah keadaan dengan memperbarui nilai untuk kunci yang sesuai. Dari sudut pandang lain, di setiap superstep, LangGraph dapat menghasilkan nilai baru untuk kunci tertentu. Dengan kata lain, untuk setiap kunci dalam keadaan, ada urutan nilai, dan dari perspektif pemrograman fungsional, fungsi `reduce` dapat diterapkan ke urutan ini. Reducer bawaan di LangGraph selalu mengganti nilai akhir dengan nilai baru. Bayangkan kita ingin melacak tindakan kustom (dihasilkan oleh node) dan membandingkan tiga opsi.

Dengan opsi pertama, node harus mengembalikan daftar sebagai nilai untuk kunci `actions`. Kami berikan contoh kode singkat hanya untuk tujuan ilustrasi, tapi kamu dapat temukan yang lengkap di Github. Jika nilai seperti itu sudah ada dalam keadaan, itu akan diganti dengan yang baru:

```python
class JobApplicationState(TypedDict):
   ...
   actions: list[str]
```

Opsi lain adalah menggunakan metode bawaan `add` dengan petunjuk tipe `Annotated`. Dengan menggunakan petunjuk tipe ini, kami memberi tahu kompiler LangGraph bahwa tipe variabel kita dalam keadaan adalah daftar string, dan ia harus menggunakan metode `add` untuk menggabungkan dua daftar (jika nilai sudah ada dalam keadaan dan node menghasilkan yang baru):

```python
from typing import Annotated, Optional
from operator import add
class JobApplicationState(TypedDict):
   ...
   actions: Annotated[list[str], add]
```

Opsi terakhir adalah menulis reducer kustom sendiri. Dalam contoh ini, kami menulis reducer kustom yang menerima tidak hanya daftar dari node (sebagai nilai baru) tetapi juga string tunggal yang akan dikonversi menjadi daftar:

```python
from typing import Annotated, Optional, Union
def my_reducer(left: list[str], right: Optional[Union[str, list[str]]]) -> list[str]:
  if right:
    return left + [right] if isinstance(right, str) else left + right
  return left
class JobApplicationState(TypedDict):
   ...
   actions: Annotated[list[str], my_reducer]
```

LangGraph memiliki beberapa reducer bawaan, dan kami juga akan tunjukkan bagaimana kamu dapat menerapkan milikmu sendiri. Salah satu yang penting adalah `add_messages`, yang memungkinkan kita menggabungkan pesan. Banyak node-mu akan menjadi agen LLM, dan LLM biasanya bekerja dengan pesan. Oleh karena itu, menurut paradigma pemrograman percakapan yang akan kita bahas lebih detail di _Bab 5_ dan _Bab 6_, kamu biasanya perlu melacak pesan-pesan ini:

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
class JobApplicationState(TypedDict):
  ...
  messages: Annotated[list[AnyMessage], add_messages]
```

Karena ini reducer yang begitu penting, ada keadaan bawaan yang dapat kamu warisi:

```python
from langgraph.graph import MessagesState
class JobApplicationState(MessagesState):
  ...
```

Sekarang, setelah kita membahas reducer, mari bicara tentang konsep penting lain untuk pengembang apa pun – cara menulis alur kerja yang dapat digunakan ulang dan modular dengan meneruskan konfigurasi ke mereka.

### Membuat graf dapat dikonfigurasi

LangGraph menyediakan API yang kuat yang memungkinkanmu membuat graf-mu dapat dikonfigurasi. Ini memungkinkanmu memisahkan parameter dari input pengguna – misalnya, untuk bereksperimen antara penyedia LLM berbeda atau meneruskan panggilan balik kustom. Sebuah node juga dapat mengakses konfigurasi dengan menerimanya sebagai argumen kedua. Konfigurasi akan diteruskan sebagai instance `RunnableConfig.`

`RunnableConfig` adalah kamus bertipe yang memberimu kendali atas pengaturan kendali eksekusi. Misalnya, kamu dapat mengontrol jumlah maksimum superstep dengan parameter `recursion_limit`. `RunnableConfig` juga memungkinkanmu meneruskan parameter kustom sebagai kamus terpisah di bawah kunci `configurable`.

Mari izinkan node kita menggunakan LLM berbeda selama pembuatan lamaran:

```python
from langchain_core.runnables.config import RunnableConfig
def generate_application(state: JobApplicationState, config: RunnableConfig):
   model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.0-flash-lite")
    print(f"...menghasilkan lamaran dengan {model_provider} dan {model_name} ...")
    return {"application": "some_fake_application", "actions": ["action2", "action3"]}
```

Sekarang mari kompilasi dan jalankan graf kita dengan konfigurasi kustom (jika kamu tidak berikan apa pun, LangGraph akan gunakan yang bawaan):

```python
res = graph.invoke({"job_description":"fake_jd"}, config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}})
print(res)
```

```
>> ...Menganalisis deskripsi pekerjaan yang diberikan ...
...menghasilkan lamaran dengan OpenAI dan OpenAI ...
{'job_description': 'fake_jd', 'is_suitable': True, 'application': 'some_fake_application', 'actions': ['action1', 'action2', 'action3']}
```

Sekarang setelah kita menetapkan cara menyusun alur kerja kompleks dengan LangGraph, mari lihat tantangan umum yang dihadapi alur kerja ini: memastikan keluaran LLM mengikuti struktur tepat yang dibutuhkan oleh komponen hilir. Penguraian keluaran yang kokoh dan penanganan kesalahan yang elegan sangat penting untuk pipa AI yang andal.

### Pembuatan keluaran terkendali

Saat kamu mengembangkan alur kerja kompleks, salah satu tugas umum yang perlu kamu selesaikan adalah memaksa LLM untuk menghasilkan keluaran yang mengikuti struktur tertentu. Ini disebut pembuatan terkendali. Dengan cara ini, itu dapat dikonsumsi secara terprogram oleh langkah-langkah berikutnya dalam alur kerja. Misalnya, kita dapat meminta LLM untuk menghasilkan JSON atau XML untuk panggilan API, mengekstrak atribut tertentu dari teks, atau menghasilkan tabel CSV. Ada banyak cara untuk mencapainya, dan kita akan mulai mengeksplorasi mereka di bab ini dan lanjutkan di [Bab 5](Chapter_5.xhtml#_idTextAnchor111). Karena LLM mungkin tidak selalu mengikuti struktur keluaran tepat, langkah berikutnya mungkin gagal, dan kamu perlu pulih dari kesalahan. Karenanya, kita juga akan mulai membahas penanganan kesalahan di bagian ini.

#### Penguraian keluaran

Penguraian keluaran penting saat mengintegrasikan LLM ke dalam alur kerja yang lebih besar, di mana langkah berikutnya membutuhkan data terstruktur daripada respons bahasa alami. Salah satu cara untuk melakukannya adalah dengan menambahkan instruksi yang sesuai ke petunjuk dan menguraikan keluaran.

Mari lihat tugas sederhana. Kami ingin mengklasifikasikan apakah deskripsi pekerjaan tertentu cocok untuk pemrogram Java junior sebagai langkah pipa kami dan, berdasarkan keputusan LLM, kami ingin melanjutkan dengan lamaran atau mengabaikan deskripsi pekerjaan khusus ini. Kita dapat mulai dengan petunjuk sederhana:

```python
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model="gemini-2.0-flash-lite")
job_description: str = ...  # masukkan JD-mu di sini
prompt_template = (
   "Diberikan deskripsi pekerjaan, putuskan apakah cocok untuk pengembang Java junior."
   "\nDESKRIPSI PEKERJAAN:\n{job_description}\n"
)
result = llm.invoke(prompt_template.format(job_description=job_description))
print(result.content)
```

```
>> Tidak, deskripsi pekerjaan ini tidak cocok untuk pengembang Java junior.\n\nAlasan utamanya:\n\n* … (keluaran dikurangi)
```

Seperti yang dapat kamu lihat, keluaran LLM adalah teks bebas, yang mungkin sulit diurai atau ditafsirkan dalam langkah pipa berikutnya. Bagaimana jika kita tambahkan instruksi spesifik ke petunjuk?

```python
prompt_template_enum = (
   "Diberikan deskripsi pekerjaan, putuskan apakah cocok untuk pengembang Java junior."
   "\nDESKRIPSI PEKERJAAN:\n{job_description}\n\nJawab hanya YA atau TIDAK."
)
result = llm.invoke(prompt_template_enum.format(job_description=job_description))
print(result.content)
```

```
>> TIDAK
```

Sekarang, bagaimana kita bisa mengurai keluaran ini? Tentu saja, langkah berikutnya kita bisa hanya melihat teks dan memiliki kondisi berdasarkan perbandingan string. Tapi itu tidak akan berhasil untuk kasus penggunaan yang lebih kompleks – misalnya, jika langkah berikutnya mengharapkan keluaran menjadi objek JSON. Untuk mengatasinya, LangChain menawarkan banyak OutputParsers yang mengambil keluaran yang dihasilkan oleh LLM dan mencoba menguraikannya ke format yang diinginkan (dengan memeriksa skema jika diperlukan) – daftar, CSV, enum, pandas DataFrame, model Pydantic, JSON, XML, dan sebagainya. Setiap parser mengimplementasikan antarmuka `BaseGenerationOutputParser`, yang memperluas antarmuka `Runnable` dengan metode tambahan `parse_result`.

Mari buat parser yang mengurai keluaran menjadi enum:

```python
from enum import Enum
from langchain.output_parsers import EnumOutputParser
from langchain_core.messages import HumanMessage
class IsSuitableJobEnum(Enum):
   YA = "YA"
   TIDAK = "TIDAK"
parser = EnumOutputParser(enum=IsSuitableJobEnum)
assert parser.invoke("TIDAK") == IsSuitableJobEnum.TIDAK
assert parser.invoke("YA\n") == IsSuitableJobEnum.YA
assert parser.invoke(" YA \n") == IsSuitableJobEnum.YA
assert parser.invoke(HumanMessage(content="YA")) == IsSuitableJobEnum.YA
```

`EnumOutputParser` mengonversi keluaran teks menjadi instance `Enum` yang sesuai. Perhatikan bahwa parser menangani keluaran seperti generasi apa pun (tidak hanya string), dan itu sebenarnya juga menghapus spasi keluaran.

> Kamu dapat temukan daftar lengkap parser di dokumentasi di [https://python.langchain.com/docs/concepts/output_parsers/](https://python.langchain.com/docs/concepts/output_parsers/), dan jika kamu butuh parser-mu sendiri, kamu selalu dapat membangun yang baru!

Sebagai langkah terakhir, mari gabungkan semuanya menjadi rantai:

```python
chain = llm | parser
result = chain.invoke(prompt_template_enum.format(job_description=job_description))
print(result)
```

```
>> TIDAK
```

Sekarang mari buat rantai ini bagian dari alur kerja LangGraph kita:

```python
class JobApplicationState(TypedDict):
   job_description: str
   is_suitable: IsSuitableJobEnum
   application: str
analyze_chain = llm | parser
def analyze_job_description(state):
   prompt = prompt_template_enum.format(job_description=state["job_description"])
   result = analyze_chain.invoke(prompt)
   return {"is_suitable": result}
def is_suitable_condition(state: JobApplicationState):
   return state["is_suitable"] == IsSuitableJobEnum.YA
builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges(
   "analyze_job_description", is_suitable_condition,
    {True: "generate_application", False: END})
builder.add_edge("generate_application", END)
```

Kami membuat dua perubahan penting. Pertama, rantai yang baru kami bangun sekarang menjadi bagian dari fungsi Python yang mewakili node `analyze_job_description`, dan itulah cara kami mengimplementasikan logika dalam node. Kedua, fungsi sisi bersyarat kami tidak mengembalikan string lagi, tapi kami menambahkan pemetaan nilai yang dikembalikan ke sisi tujuan ke fungsi `add_conditional_edges`, dan itu adalah contoh bagaimana kamu dapat mengimplementasikan percabangan alur kerja-mu.

Mari luangkan waktu untuk membahas bagaimana menangani kesalahan potensial jika penguraian kita gagal!

#### Penanganan kesalahan

Pengelolaan kesalahan yang efektif sangat penting dalam alur kerja LangChain apa pun, termasuk saat menangani kegagalan alat (yang akan kita eksplorasi di [Bab 5](Chapter_5.xhtml#_idTextAnchor111) saat kita membahas alat). Saat mengembangkan aplikasi LangChain, ingat bahwa kegagalan dapat terjadi di tahap apa pun:

- Panggilan API ke model fondasi mungkin gagal
- LLM mungkin menghasilkan keluaran yang tidak terduga
- Layanan eksternal bisa menjadi tidak tersedia

Salah satu pendekatan yang mungkin adalah menggunakan mekanisme Python dasar untuk menangkap pengecualian, mencatatnya untuk analisis lebih lanjut, dan melanjutkan alur kerja-mu baik dengan membungkus pengecualian sebagai teks atau dengan mengembalikan nilai default. Jika rantai LangChain-mu memanggil beberapa fungsi Python kustom, pikirkan tentang penanganan pengecualian yang tepat. Hal yang sama berlaku untuk node LangGraph-mu.

Pencatatan sangat penting, terutama saat kamu mendekati penyebaran produksi. Pencatatan yang tepat memastikan bahwa pengecualian tidak luput dari perhatian, memungkinkanmu memantau kejadiannya. Alat observabilitas modern menyediakan mekanisme pemberitahuan yang mengelompokkan kesalahan serupa dan memberi tahu kamu tentang masalah yang sering terjadi.

Mengonversi pengecualian ke teks memungkinkan alur kerja-mu melanjutkan eksekusi sambil memberikan konteks berharga kepada LLM hilir tentang apa yang salah dan jalur pemulihan potensial. Berikut contoh sederhana bagaimana kamu dapat mencatat pengecualian tetapi melanjutkan mengeksekusi alur kerja-mu dengan tetap pada perilaku default:

```python
import logging
logger = logging.getLogger(__name__)
llms = {
   "palsu": fake_llm,
   "Google": llm
}
def analyze_job_description(state, config: RunnableConfig):
   try:
     llm = config["configurable"].get("model_provider", "Google")
      llm = llms[model_provider]
      analyze_chain = llm | parser
      prompt = prompt_template_enum.format(job_description=job_description)
      result = analyze_chain.invoke(prompt)
      return {"is_suitable": result}
    except Exception as e:
      logger.error(f"Pengecualian {e} terjadi saat mengeksekusi analyze_job_description")
      return {"is_suitable": False}
```

Untuk menguji penanganan kesalahan kita, kita perlu mensimulasikan kegagalan LLM. LangChain memiliki beberapa kelas `FakeChatModel` yang membantumu menguji rantaimu:

- `GenericFakeChatModel` mengembalikan pesan berdasarkan iterator yang disediakan
- `FakeChatModel` selalu mengembalikan string `"fake_response"`
- `FakeListChatModel` mengambil daftar pesan dan mengembalikannya satu per satu pada setiap panggilan

Mari buat LLM palsu yang gagal setiap kali kedua:

```python
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
class MessagesIterator:
   def __init__(self):
       self._count = 0
   def __iter__(self):
       return self
   def __next__(self):
       self._count += 1
       if self._count % 2 == 1:
           raise ValueError("Ada yang salah")
       return AIMessage(content="False")
fake_llm = GenericFakeChatModel(messages=MessagesIterator())
```

Saat kami berikan ini ke graf kami (contoh kode lengkap tersedia di repo GitHub kami), kita dapat lihat bahwa alur kerja berlanjut meskipun menemui pengecualian:

```python
res = graph.invoke({"job_description":"fake_jd"}, config={"configurable": {"model_provider": "fake"}})
print(res)
```

```
>> ERROR:__main__:Pengecualian Diharapkan Runnable, callable atau dict. Sebagai gantinya mendapat tipe yang tidak didukung: <class 'str'> terjadi saat mengeksekusi analyze_job_description
{'job_description': 'fake_jd', 'is_suitable': False}
```

Saat kesalahan terjadi, terkadang mencoba lagi membantu. LLM memiliki sifat non-deterministik, dan upaya berikutnya mungkin berhasil; juga, jika kamu menggunakan API pihak ketiga, berbagai kegagalan mungkin terjadi di sisi penyedia. Mari bahas bagaimana mengimplementasikan percobaan ulang yang tepat dengan LangGraph.

##### Percobaan ulang

Ada tiga pendekatan percobaan ulang berbeda, masing-masing cocok untuk skenario berbeda:

- Percobaan ulang generik dengan Runnable
- Kebijakan percobaan ulang khusus node
- Perbaikan keluaran semantik

Mari lihat ini secara bergantian, mulai dengan percobaan ulang generik yang tersedia untuk setiap `Runnable.`

Kamu dapat mencoba ulang `Runnable` atau node LangGraph apa pun menggunakan mekanisme bawaan:

```python
fake_llm_retry = fake_llm.with_retry(
   retry_if_exception_type=(ValueError,),
   wait_exponential_jitter=True,
   stop_after_attempt=2,
)
analyze_chain_fake_retries = fake_llm_retry | parser
```

Dengan LangGraph, kamu juga dapat menjelaskan percobaan ulang spesifik untuk setiap node. Misalnya, mari coba ulang node `analyze_job_description` kita dua kali dalam kasus `ValueError`:

```python
from langgraph.pregel import RetryPolicy
builder.add_node(
  "analyze_job_description", analyze_job_description,
  retry=RetryPolicy(retry_on=ValueError, max_attempts=2))
```

Komponen yang kamu gunakan, sering dikenal sebagai blok bangunan, mungkin memiliki mekanisme percobaan ulang sendiri yang mencoba memperbaiki masalah secara algoritmik dengan memberikan input tambahan ke LLM tentang apa yang salah. Misalnya, banyak model chat di LangChain memiliki percobaan ulang sisi klien pada kesalahan sisi server tertentu.

ChatAnthropic memiliki parameter `max_retries` yang dapat kamu definisikan baik per instance atau per permintaan. Contoh bagus lain dari blok bangunan yang lebih canggih adalah mencoba pulih dari kesalahan penguraian. Mencoba ulang langkah penguraian tidak akan membantu karena biasanya kesalahan penguraian terkait dengan keluaran LLM yang tidak lengkap. Bagaimana jika kita coba ulang langkah generasi dan berharap yang terbaik, atau sebenarnya memberi petunjuk LLM tentang apa yang salah? Itulah tepatnya yang dilakukan `RetryWithErrorOutputParser`.

![Gambar 3.3: Menambahkan mekanisme percobaan ulang ke rantai yang memiliki beberapa langkah](Images/B32363_03_03.png)

Untuk menggunakan `RetryWithErrorOutputParser`, kita perlu menginisialisasinya terlebih dahulu dengan LLM (digunakan untuk memperbaiki keluaran) dan parser kita. Lalu, jika penguraian kita gagal, kita jalankan dan berikan petunjuk awal kita (dengan semua parameter yang diganti), respons yang dihasilkan, dan kesalahan penguraian:

```python
from langchain.output_parsers import RetryWithErrorOutputParser
fix_parser = RetryWithErrorOutputParser.from_llm(
  llm=llm, # berikan llm di sini
  parser=parser, # parser asli-mu yang gagal
  prompt=retry_prompt, # parameter opsional, kamu dapat mendefinisikan ulang petunjuk default
)
fixed_output = fix_parser.parse_with_prompt(
  completion=original_response, prompt_value=original_prompt)
```

Kita dapat membaca kode sumber di GitHub untuk lebih memahami apa yang terjadi, tetapi pada dasarnya, itu adalah contoh kode semu tanpa terlalu banyak detail. Kami ilustrasikan bagaimana kita dapat meneruskan kesalahan penguraian dan keluaran asli yang menyebabkan kesalahan ini kembali ke LLM dan memintanya memperbaiki masalah:

```python
prompt = """
Petunjuk: {prompt} Penyelesaian: {completion} Di atas, Penyelesaian tidak memenuhi batasan yang diberikan dalam Petunjuk. Detail: {error} Silakan coba lagi:
"""
retry_chain = prompt | llm | StrOutputParser()
# coba mengurai penyelesaian dengan parser yang disediakan
parser.parse(completion)
# jika gagal, tangkap kesalahan dan coba pulih maksimal upaya percobaan ulang
completion = retry_chain.invoke(original_prompt, completion, error)
```

Kami memperkenalkan `StrOutputParser` di [Bab 2](Chapter_2.xhtml#_idTextAnchor025) untuk mengonversi keluaran ChatModel dari AIMessage menjadi string sehingga kita dapat dengan mudah meneruskannya ke langkah berikutnya dalam rantai.

Hal lain yang perlu diingat adalah bahwa blok bangunan LangChain memungkinkanmu mendefinisikan ulang parameter, termasuk petunjuk default. Kamu selalu dapat memeriksanya di Github; terkadang ide bagus untuk menyesuaikan petunjuk default untuk alur kerja-mu.

> Kamu dapat baca tentang parser perbaikan keluaran lain yang tersedia di sini: [https://python.langchain.com/docs/how_to/output_parser_retry/](https://python.langchain.com/docs/how_to/output_parser_retry/).

##### Cadangan

Dalam pengembangan perangkat lunak, **cadangan** adalah program alternatif yang memungkinkanmu pulih jika program dasar-mu gagal. LangChain memungkinkanmu mendefinisikan cadangan pada tingkat `Runnable`. Jika eksekusi gagal, rantai alternatif dipicu dengan parameter input yang sama. Misalnya, jika LLM yang kamu gunakan tidak tersedia untuk periode singkat, rantaimu akan secara otomatis beralih ke yang berbeda yang menggunakan penyedia alternatif (dan mungkin petunjuk berbeda).

Model palsu kita gagal setiap kali kedua, jadi mari tambahkan cadangan ke sana. Itu hanya lambda yang mencetak pernyataan. Seperti yang dapat kita lihat, setiap kali kedua, cadangan dieksekusi:

```python
from langchain_core.runnables import RunnableLambda
chain_fallback = RunnableLambda(lambda _: print("menjalankan cadangan"))
chain = fake_llm | RunnableLambda(lambda _: print("menjalankan rantai utama"))
chain_with_fb = chain.with_fallbacks([chain_fallback])
chain_with_fb.invoke("test")
chain_with_fb.invoke("test")
```

```
>> menjalankan cadangan
menjalankan rantai utama
```

Menghasilkan hasil kompleks yang dapat mengikuti templat tertentu dan dapat diurai dengan andal disebut generasi terstruktur (atau generasi terkendali). Ini dapat membantu membangun alur kerja yang lebih kompleks, di mana keluaran dari satu langkah yang digerakkan LLM dapat dikonsumsi oleh langkah terprogram lainnya. Kami akan angkat ini lagi lebih detail di _Bab 5_ dan _Bab 6_.

Petunjuk yang kamu kirim ke LLM adalah salah satu blok bangunan terpenting dari alur kerja-mu. Karenanya, mari bahas beberapa dasar rekayasa petunjuk berikutnya dan lihat bagaimana mengatur petunjupmu dengan LangChain.

## Rekayasa petunjuk

Mari lanjutkan dengan melihat ke rekayasa petunjuk dan mengeksplorasi berbagai sintaks LangChain terkait dengannya. Tapi pertama, mari bahas bagaimana rekayasa petunjuk berbeda dari desain petunjuk. Istilah-istilah ini terkadang digunakan secara bergantian, dan itu menciptakan tingkat kebingungan tertentu. Seperti yang kita bahas di [Bab 1](Chapter_1.xhtml#_idTextAnchor000), salah satu penemuan besar tentang LLM adalah bahwa mereka memiliki kemampuan adaptasi domain dengan _pembelajaran dalam konteks_. Seringkali cukup mendeskripsikan tugas yang ingin kita lakukan dalam bahasa alami, dan meskipun LLM tidak dilatih pada tugas khusus ini, ia bekerja sangat baik. Tapi seperti yang dapat kita bayangkan, ada banyak cara mendeskripsikan tugas yang sama, dan LLM sensitif terhadap ini. Meningkatkan petunjuk kita (atau templat petunjuk, lebih spesifik) untuk meningkatkan kinerja pada tugas tertentu disebut rekayasa petunjuk. Namun, mengembangkan petunjuk yang lebih universal yang memandu LLM untuk menghasilkan respons yang umumnya lebih baik pada kumpulan tugas yang luas disebut desain petunjuk.

Ada berbagai macam teknik rekayasa petunjuk berbeda. Kami tidak akan membahas banyak darinya secara detail di bagian ini, tapi kami akan menyentuh hanya beberapa darinya untuk mengilustrasikan kemampuan kunci LangChain yang akan memungkinkanmu membangun petunjuk apa pun yang kamu inginkan.

> Kamu dapat temukan ikhtisar bagus tentang taksonomi petunjuk di makalah _The Prompt Report: A Systematic Survey of Prompt Engineering Techniques_, diterbitkan oleh Sander Schulhoff dan rekan: [https://arxiv.org/abs/2406.06608](https://arxiv.org/abs/2406.06608).

### Templat petunjuk

Apa yang kami lakukan di [Bab 2](Chapter_2.xhtml#_idTextAnchor025) disebut _zero-shot prompting_. Kami membuat templat petunjuk yang berisi deskripsi setiap tugas. Saat kami menjalankan alur kerja, kami mengganti nilai tertentu dari templat petunjuk ini dengan argumen waktu proses. LangChain memiliki beberapa abstraksi yang sangat berguna untuk membantu dengan itu.

Di [Bab 2](Chapter_2.xhtml#_idTextAnchor025), kami memperkenalkan `PromptTemplate`, yang merupakan `RunnableSerializable`. Ingat bahwa itu mengganti templat string selama pemanggilan – misalnya, kamu dapat membuat templat berdasarkan f-string dan menambahkan rantaimu, dan LangChain akan meneruskan parameter dari input, menggantinya dalam templat, dan meneruskan string ke langkah berikutnya dalam rantai:

```python
from langchain_core.output_parsers import StrOutputParser
lc_prompt_template = PromptTemplate.from_template(prompt_template)
chain = lc_prompt_template | llm | StrOutputParser()
chain.invoke({"job_description": job_description})
```

Untuk model chat, input tidak hanya dapat berupa string tetapi juga daftar `messages` – misalnya, pesan sistem diikuti oleh riwayat percakapan. Oleh karena itu, kita juga dapat membuat templat yang menyiapkan daftar pesan, dan templat itu sendiri dapat dibuat berdasarkan daftar pesan atau templat pesan, seperti dalam contoh ini:

```python
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
msg_template = HumanMessagePromptTemplate.from_template(
  prompt_template)
msg_example = msg_template.format(job_description="fake_jd")
chat_prompt_template = ChatPromptTemplate.from_messages([
  SystemMessage(content="Kamu adalah asisten yang membantu."),
  msg_template])
chain = chat_prompt_template | llm | StrOutputParser()
chain.invoke({"job_description": job_description})
```

Kamu juga dapat melakukan hal yang sama dengan lebih nyaman tanpa menggunakan templat petunjuk chat tetapi dengan mengirimkan tupel (hanya karena lebih cepat dan lebih nyaman terkadang) dengan tipe pesan dan string bertemplat sebagai gantinya:

```python
chat_prompt_template = ChatPromptTemplate.from_messages(
   [("system", "Kamu adalah asisten yang membantu."),
    ("human", prompt_template)])
```

Konsep penting lain adalah _placeholder_. Ini mengganti variabel dengan daftar pesan yang disediakan secara real time. Kamu dapat menambahkan placeholder ke petunjupmu dengan menggunakan petunjuk `placeholder`, atau menambahkan `MessagesPlaceholder`:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
chat_prompt_template = ChatPromptTemplate.from_messages(
   [("system", "Kamu adalah asisten yang membantu."),
    ("placeholder", "{history}"),
    # sama dengan MessagesPlaceholder("history"),
    ("human", prompt_template)])
len(chat_prompt_template.invoke({"job_description": "fake", "history": [("human", "hai!"), ("ai", "hai!")]}).messages)
```

```
>> 4
```

Sekarang input kita terdiri dari empat pesan – pesan sistem, dua pesan riwayat yang kami berikan, dan satu pesan manusia dari petunjuk bertemplat. Contoh terbaik menggunakan placeholder adalah untuk menginput riwayat chat, tapi kita akan lihat yang lebih canggih nanti di buku ini saat kita bicarakan bagaimana LLM berinteraksi dengan dunia eksternal atau bagaimana LLM berbeda berkoordinasi bersama dalam pengaturan multi-agen.

### Zero-shot vs. few-shot prompting

Seperti yang telah kita bahas, hal pertama yang ingin kita eksperimenkan adalah meningkatkan deskripsi tugas itu sendiri. Deskripsi tugas tanpa contoh solusi disebut **zero-shot** prompting, dan ada banyak trik yang dapat kamu coba.

Apa yang biasanya bekerja baik adalah menetapkan LLM peran tertentu (misalnya, "_Kamu adalah asisten perusahaan berguna yang bekerja untuk XXX perusahaan Fortune-500_") dan memberikan beberapa instruksi tambahan (misalnya, apakah LLM harus kreatif, ringkas, atau faktual). Ingat bahwa LLM telah melihat berbagai data dan mereka dapat melakukan tugas berbeda, dari menulis buku fantasi hingga menjawab pertanyaan penalaran kompleks. Tapi tujuanmu adalah menginstruksikan mereka, dan jika kamu ingin mereka tetap pada fakta, lebih baik berikan instruksi yang sangat spesifik sebagai bagian dari profil peran mereka. Untuk model chat, pengaturan peran seperti itu biasanya terjadi melalui pesan sistem (tapi ingat bahwa, bahkan untuk model chat, semuanya digabungkan menjadi satu petunjuk input yang diformat di sisi server).

Panduan petunjuk Gemini merekomendasikan bahwa setiap petunjuk harus memiliki empat bagian: persona, tugas, konteks relevan, dan format yang diinginkan. Ingat bahwa penyedia model berbeda mungkin memiliki rekomendasi berbeda tentang penulisan atau pemformatan petunjuk, karenanya jika kamu memiliki petunjuk kompleks, selalu periksa dokumentasi penyedia model, evaluasi kinerja alur kerja-mu sebelum beralih ke penyedia model baru, dan sesuaikan petunjuk sesuai kebutuhan jika diperlukan. Jika kamu ingin menggunakan banyak penyedia model dalam produksi, kamu mungkin berakhir dengan banyak templat petunjuk dan memilihnya secara dinamis berdasarkan penyedia model.

Peningkatan besar lain dapat dengan memberikan LLM beberapa contoh tugas khusus ini sebagai pasangan input-output sebagai bagian dari petunjuk. Ini disebut few-shot prompting. Biasanya, few-shot prompting sulit digunakan dalam skenario yang membutuhkan input panjang (seperti RAG, yang akan kita bicarakan di bab berikutnya) tapi itu masih sangat berguna untuk tugas dengan petunjuk yang relatif pendek, seperti klasifikasi, ekstraksi, dll.

Tentu saja, kamu selalu dapat mengkodekan contoh dalam templat petunjuk itu sendiri, tapi ini membuatnya sulit untuk mengelolanya seiring sistemmu tumbuh. Cara yang lebih baik mungkin adalah menyimpan contoh dalam file terpisah di disk atau dalam database dan memuatnya ke petunjupmu.

#### Merantai petunjuk bersama

Seiring petunjupmu menjadi lebih canggih, mereka cenderung tumbuh dalam ukuran dan kompleksitas. Skenario umum adalah memformat sebagian petunjupmu, dan kamu dapat melakukan ini baik dengan substitusi string atau fungsi. Yang terakhir relevan jika beberapa bagian petunjupmu bergantung pada variabel yang berubah secara dinamis (misalnya, tanggal saat ini, nama pengguna, dll.). Di bawah, kamu dapat temukan contoh substitusi parsial dalam templat petunjuk:

```python
system_template = PromptTemplate.from_template("a: {a} b: {b}")
system_template_part = system_template.partial(
   a="a" # kamu juga dapat memberikan fungsi di sini
)
print(system_template_part.invoke({"b": "b"}).text)
```

```
>> a: a b: b
```

Cara lain untuk membuat petunjupmu lebih dapat dikelola adalah dengan membaginya menjadi beberapa bagian dan merantaikannya bersama:

```python
system_template_part1 = PromptTemplate.from_template("a: {a}")
system_template_part2 = PromptTemplate.from_template("b: {b}")
system_template = system_template_part1 + system_template_part2
print(system_template_part.invoke({"a": "a", "b": "b"}).text)
```

```
>> a: a b: b
```

Kamu juga dapat membangun substitusi yang lebih kompleks dengan menggunakan kelas `langchain_core.prompts.PipelinePromptTemplate`. Selain itu, kamu dapat meneruskan templat ke `ChatPromptTemplate` dan mereka akan secara otomatis disusun bersama:

```python
system_prompt_template = PromptTemplate.from_template("a: {a} b: {b}")
chat_prompt_template = ChatPromptTemplate.from_messages(
   [("system", system_prompt_template.template),
    ("human", "hai"),
    ("ai", "{c}")])
messages = chat_prompt_template.invoke({"a": "a", "b": "b", "c": "c"}).messages
print(len(messages))
print(messages[0].content)
```

```
>> 3
a: a b: b
```

#### Few-shot prompting dinamis

Seiring jumlah contoh yang digunakan dalam petunjuk few-shot-mu terus tumbuh, kamu mungkin membatasi jumlah contoh yang akan diteruskan ke substitusi templat petunjuk tertentu. Kami memilih contoh untuk setiap input – dengan mencari contoh yang mirip dengan input pengguna (kita akan bicara lebih banyak tentang kemiripan semantik dan embedding di [Bab 4](Chapter_4.xhtml#_idTextAnchor068)), membatasinya berdasarkan panjang, mengambil yang terbaru, dll.

![Gambar 3.4: Contoh alur kerja dengan pengambilan dinamis contoh untuk diteruskan ke petunjuk few-shot](Images/B32363_03_04.png)

Ada beberapa pemilih yang sudah dibangun di bawah `langchain_core.example_selectors`. Kamu dapat langsung meneruskan instance pemilih contoh ke instance `FewShotPromptTemplate` selama instansiasi.

### Rantai Pikiran

Tim Google Research memperkenalkan teknik **Rantai-Pikiran** (**Chain-of-Thought, CoT**) awal tahun 2022. Mereka menunjukkan bahwa modifikasi relatif sederhana pada petunjuk yang mendorong model untuk menghasilkan langkah-langkah penalaran perantara langkah-demi-langkah secara signifikan meningkatkan kinerja LLM pada tugas penalaran simbolik kompleks, akal sehat, dan matematika. Peningkatan kinerja seperti itu telah direplikasi berkali-kali sejak itu.

> Kamu dapat baca makalah asli memperkenalkan CoT, _Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_, diterbitkan oleh Jason Wei dan rekan: [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).

Ada modifikasi berbeda dari CoT prompting, dan karena memiliki keluaran panjang, biasanya, petunjuk CoT adalah zero-shot. Kamu menambahkan instruksi yang mendorong LLM untuk berpikir tentang masalah terlebih dahulu alih-alih segera menghasilkan token yang mewakili jawaban. Contoh CoT yang sangat sederhana adalah hanya menambahkan ke templat petunjupmu sesuatu seperti "Mari berpikir langkah demi langkah."

Ada berbagai petunjuk CoT yang dilaporkan di makalah berbeda. Kamu juga dapat menjelajahi templat CoT yang tersedia di LangSmith. Untuk tujuan pembelajaran kami, mari gunakan petunjuk CoT dengan contoh few-shot:

```python
from langchain import hub
math_cot_prompt = hub.pull("arietem/math_cot")
cot_chain = math_cot_prompt | llm | StrOutputParser()
print(cot_chain.invoke("Selesaikan persamaan 2*x+5=15"))
```

```
>> Jawaban: Mari berpikir langkah demi langkah
Kurangi 5 dari kedua sisi:
2x + 5 - 5 = 15 - 5
2x = 10
Bagi kedua sisi dengan 2:
2x / 2 = 10 / 2
x = 5
```

Kami menggunakan petunjuk dari LangSmith Hub – kumpulan artefak pribadi dan publik yang dapat kamu gunakan dengan LangChain. Kamu dapat jelajahi petunjuk itu sendiri di sini: [https://smith.langchain.com/hub.](https://smith.langchain.com/hub.).

Dalam praktik, kamu mungkin ingin membungkus panggilan CoT dengan langkah ekstraksi untuk memberikan jawaban ringkas kepada pengguna. Misalnya, mari kita jalankan `cot_chain` terlebih dahulu dan kemudian teruskan keluarannya (harap dicatat bahwa kami meneruskan kamus dengan `question` awal dan `cot_output` ke langkah berikutnya) ke LLM yang akan menggunakan petunjuk untuk membuat jawaban akhir berdasarkan penalaran CoT:

```python
from operator import itemgetter
parse_prompt_template = (
   "Diberikan pertanyaan awal dan jawaban lengkap, "
   "ekstrak jawaban ringkas. Jangan berasumsi apa pun dan "
   "hanya gunakan jawaban lengkap yang disediakan.\n\nPERTANYAAN:\n{question}\n"
   "JAWABAN LENGKAP:\n{full_answer}\n\nJAWABAN RINGKAS:\n"
)
parse_prompt = PromptTemplate.from_template(
   parse_prompt_template
)
final_chain = (
 { "full_answer": itemgetter("question") | cot_chain,
   "question": itemgetter("question"),
 }
 | parse_prompt
 | llm
 | StrOutputParser()
)
print(final_chain.invoke({"question": "Selesaikan persamaan 2*x+5=15"}))
```

```
>> 5
```
