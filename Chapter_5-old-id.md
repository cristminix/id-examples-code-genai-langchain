# Membangun Agen Cerdas

Seiring adopsi AI generatif tumbuh, kita mulai menggunakan LLM (Large Language Model) untuk tugas yang lebih terbuka dan kompleks yang membutuhkan pengetahuan tentang peristiwa terbaru atau interaksi dengan dunia. Inilah yang umumnya disebut aplikasi agen. Kita akan mendefinisikan apa itu agen nanti di bab ini, tetapi Anda mungkin pernah melihat frasa yang beredar di media: _2025 adalah tahun AI agen_. Misalnya, dalam RE-Bench benchmark yang baru diperkenalkan yang terdiri dari tugas terbuka kompleks, agen AI mengungguli manusia dalam beberapa pengaturan (misalnya, dengan anggaran berpikir 30 menit) atau pada beberapa kelas tugas tertentu (seperti menulis kernel Triton).

Untuk memahami bagaimana kemampuan agen ini dibangun dalam praktik, kita akan mulai dengan membahas pemanggilan alat dengan LLM dan bagaimana hal itu diimplementasikan pada LangChain. Kita akan melihat secara detail pola ReACT, dan bagaimana LLM dapat menggunakan alat untuk berinteraksi dengan lingkungan eksternal dan meningkatkan kinerjanya pada tugas tertentu. Kemudian, kita akan menyentuh bagaimana alat didefinisikan di LangChain, dan alat bawaan apa yang tersedia. Kita juga akan berbicara tentang mengembangkan alat kustom sendiri, menangani kesalahan, dan menggunakan kemampuan pemanggilan alat lanjutan. Sebagai contoh praktis, kita akan melihat cara menghasilkan keluaran terstruktur dengan LLM menggunakan alat versus memanfaatkan kemampuan bawaan yang ditawarkan oleh penyedia model.

Akhirnya, kita akan berbicara tentang apa itu agen dan melihat pola yang lebih canggih dalam membangun agen dengan LangGraph sebelum kita kemudian mengembangkan agen ReACT pertama kita dengan LangGraph—sebuah agen penelitian yang mengikuti pola rencana-dan-selesaikan dan menggunakan alat seperti pencarian web, _arXiv_, dan _Wikipedia_.

Singkatnya, topik berikut akan dibahas dalam bab ini:

- Apa itu alat?
- Mendefinisikan alat LangChain bawaan dan alat kustom
- Kemampuan pemanggilan alat lanjutan
- Menggabungkan alat ke dalam alur kerja
- Apa itu agen?

<div class="note">
Anda dapat menemukan kode untuk bab ini di direktori `chapter5/` repositori GitHub buku. Silakan kunjungi [https://github.com/benman1/generative_ai_with_langchain/tree/second_edition](https://github.com/benman1/generative_ai_with_langchain/tree/second_edition) untuk pembaruan terbaru.

Lihat [Bab 2](Chapter_2.xhtml#_idTextAnchor025) untuk instruksi penyiapan. Jika Anda memiliki pertanyaan atau mengalami masalah saat menjalankan kode, silakan buat isu di GitHub atau bergabung dalam diskusi di Discord di [https://packt.link/lang](https://packt.link/lang).

</div>

Mari kita mulai dengan alat. Daripada langsung menyelami definisi agen, lebih membantu untuk pertama-tama menjelajahi bagaimana peningkatan LLM dengan alat sebenarnya bekerja dalam praktik. Dengan berjalan melalui ini langkah demi langkah, Anda akan melihat bagaimana integrasi ini membuka kemampuan baru. Jadi, apa sebenarnya alat itu, dan bagaimana mereka memperluas apa yang dapat dilakukan LLM?

## Apa itu alat?

LLM dilatih pada data korpus umum yang sangat besar (seperti data web dan buku), yang memberi mereka pengetahuan luas tetapi membatasi efektivitasnya dalam tugas yang membutuhkan pengetahuan khusus domain atau terkini. Namun, karena LLM baik dalam penalaran, mereka dapat berinteraksi dengan lingkungan eksternal melalui alat—API atau antarmuka yang memungkinkan model berinteraksi dengan dunia luar. Alat ini memungkinkan LLM melakukan tugas tertentu dan menerima umpan balik dari dunia luar.

Saat menggunakan alat, LLM melakukan tiga tugas generasi spesifik:

1. Pilih alat yang akan digunakan dengan menghasilkan token khusus dan nama alat.
2. Hasilkan payload yang akan dikirim ke alat.
3. Hasilkan respons kepada pengguna berdasarkan pertanyaan awal dan riwayat interaksi dengan alat (untuk jalankan spesifik ini).

Sekarang saatnya mencari tahu bagaimana LLM memanggil alat dan bagaimana kita dapat membuat LLM sadar alat. Pertimbangkan pertanyaan yang agak buatan tetapi ilustratif: _Apa akar kuadrat dari usia presiden AS saat ini dikalikan 132_? Pertanyaan ini menyajikan dua tantangan spesifik:

- Ini merujuk informasi terkini (per Maret 2025) yang kemungkinan berada di luar data pelatihan model.
- Ini membutuhkan perhitungan matematis yang tepat yang mungkin tidak dapat dijawab dengan benar oleh LLM hanya dengan generasi token autoregresif.

Alih-alih memaksa LLM menghasilkan jawaban hanya berdasarkan pengetahuan internalnya, kita akan beri LLM akses ke dua alat: mesin pencari dan kalkulator. Kami mengharapkan model untuk menentukan alat mana yang dibutuhkan (jika ada) dan bagaimana menggunakannya.

Untuk kejelasan, mari kita mulai dengan pertanyaan yang lebih sederhana dan meniru alat kita dengan membuat fungsi dummy yang selalu memberikan respons yang sama. Nanti di bab ini, kita akan mengimplementasikan alat yang berfungsi penuh dan memanggilnya:

```python
question = "how old is the US president?"
raw_prompt_template = (
  "You have access to search engine that provides you an "
  "information about fresh events and news given the query. "
  "Given the question, decide whether you need an additional "
  "information from the search engine (reply with 'SEARCH: "
  "<generated query>' or you know enough to answer the user "
  "then reply with 'RESPONSE <final response>').\n"
  "Now, act to answer a user question:\n{QUESTION}"
)
prompt_template = PromptTemplate.from_template(raw_prompt_template)
result = (prompt_template | llm).invoke(question)
print(result,response)
```

```
>> SEARCH: current age of US president
```

Mari pastikan bahwa ketika LLM memiliki cukup pengetahuan internal, ia menjawab langsung kepada pengguna:

```python
question1 = "What is the capital of Germany?"
result = (prompt_template | llm).invoke(question1)
print(result,response)
```

```
>> RESPONSE: Berlin
```

Akhirnya, mari beri model keluaran alat dengan memasukkannya ke dalam prompt:

```python
query = "age of current US president"
search_result = (
   "Donald Trump ' Age 78 years June 14, 1946\n"
   "Donald Trump 45th and 47th U.S. President Donald John Trump is an American "
   "politician, media personality, and businessman who has served as the 47th "
   "president of the United States since January 20, 2025. A member of the "
   "Republican Party, he previously served as the 45th president from 2017 to 2021. Wikipedia"
)
raw_prompt_template = (
 "You have access to search engine that provides you an "
 "information about fresh events and news given the query. "
 "Given the question, decide whether you need an additional "
 "information from the search engine (reply with 'SEARCH: "
 "<generated query>' or you know enough to answer the user "
 "then reply with 'RESPONSE <final response>').\n"
 "Today is {date}."
 "Now, act to answer a user question and "
 "take into account your previous actions:\n"
 "HUMAN: {question}\n"
 "AI: SEARCH: {query}\n"
 "RESPONSE FROM SEARCH: {search_result}\n"
)
prompt_template = PromptTemplate.from_template(raw_prompt_template)
result = (prompt_template | llm).invoke(
  {"question": question, "query": query, "search_result": search_result,
   "date": "Feb 2025"})
print(result.content)
```

```
>>  RESPONSE: The current US President, Donald Trump, is 78 years old.
```

Sebagai pengamatan terakhir, jika hasil pencarian tidak berhasil, LLM akan mencoba menyempurnakan kueri:

```python
query = "current US president"
search_result = (
   "Donald Trump 45th and 47th U.S."
)
result = (prompt_template | llm).invoke(
  {"question": question, "query": query,
   "search_result": search_result, "date": "Feb 2025"})
print(result.content)
```

```
>>  SEARCH: Donald Trump age
```

Dengan itu, kita telah mendemonstrasikan bagaimana pemanggilan alat bekerja. Harap diperhatikan bahwa kami telah memberikan contoh prompt untuk tujuan demonstrasi saja. LLM dasar lain mungkin memerlukan beberapa rekayasa prompt, dan prompt kami hanyalah ilustrasi. Dan kabar baiknya: menggunakan alat lebih mudah daripada yang terlihat dari contoh-contoh ini!

Seperti yang dapat Anda catat, kami menjelaskan semuanya dalam prompt kami, termasuk deskripsi alat dan format pemanggilan alat. Saat ini, sebagian besar LLM menyediakan API yang lebih baik untuk pemanggilan alat karena LLM modern dilatih pasca pada kumpulan data yang membantu mereka unggul dalam tugas seperti itu. Pembuat LLM tahu bagaimana kumpulan data itu dibangun. Itulah mengapa, biasanya, Anda tidak memasukkan deskripsi alat sendiri dalam prompt; Anda hanya memberikan prompt dan deskripsi alat sebagai argumen terpisah, dan mereka digabungkan menjadi satu prompt di sisi penyedia. Beberapa LLM sumber terbuka yang lebih kecil mengharapkan deskripsi alat menjadi bagian dari prompt mentah, tetapi mereka mengharapkan format yang terdefinisi dengan baik.

LangChain memudahkan pengembangan pipa di mana LLM memanggil alat yang berbeda dan menyediakan akses ke banyak alat bawaan yang membantu. Mari kita lihat bagaimana penanganan alat bekerja dengan LangChain.

## Alat di LangChain

Dengan sebagian besar LLM modern, untuk menggunakan alat, Anda dapat memberikan daftar deskripsi alat sebagai argumen terpisah. Seperti biasa di LangChain, setiap implementasi integrasi tertentu memetakan antarmuka ke API penyedia. Untuk alat, ini terjadi melalui argumen `tools` ke metode `invoke` (dan beberapa metode berguna lainnya seperti `bind_tools` dan lainnya, seperti yang akan kita pelajari di bab ini).

Saat mendefinisikan alat, kita perlu menentukan skemanya dalam format OpenAPI. Kami memberikan _judul_ dan _deskripsi_ alat dan juga menentukan parameternya (setiap parameter memiliki _tipe_, _judul_, dan _deskripsi_). Kami dapat mewarisi skema seperti itu dari berbagai format, yang diterjemahkan LangChain ke dalam format OpenAPI. Saat kita melalui beberapa bagian berikutnya, kami akan mengilustrasikan bagaimana kita dapat melakukan ini dari fungsi, docstring, definisi Pydantic, atau dengan mewarisi dari kelas `BaseTool` dan memberikan deskripsi secara langsung. Untuk LLM, alat adalah apa pun yang memiliki spesifikasi OpenAPI—dengan kata lain, alat dapat dipanggil oleh mekanisme eksternal.

LLM sendiri tidak memperdulikan mekanisme ini, ia hanya menghasilkan instruksi untuk kapan dan bagaimana memanggil alat. Untuk LangChain, alat juga adalah sesuatu yang dapat dipanggil (dan kita akan lihat nanti bahwa alat diwarisi dari `Runnables`) ketika kita menjalankan program kita.

Kata-kata yang Anda gunakan di bidang _judul_ dan _deskripsi_ sangat penting, dan Anda dapat memperlakukannya sebagai bagian dari latihan rekayasa prompt. Kata-kata yang lebih baik membantu LLM membuat keputusan yang lebih baik tentang kapan dan bagaimana memanggil alat tertentu. Harap diperhatikan bahwa untuk alat yang lebih kompleks, menulis skema seperti ini bisa menjadi membosankan, dan kita akan melihat cara yang lebih sederhana untuk mendefinisikan alat nanti di bab ini:

```python
search_tool = {
   "title": "google_search",
    "description": "Returns about fresh events and news from Google Search engine based on a query",
   "type": "object",
   "properties": {
       "query": {
           "description": "Search query to be sent to the search engine",
           "title": "search_query",
           "type": "string"},
   },
   "required": ["query"]
}
result = llm.invoke(question, tools=[search_tool])
```

Jika kita memeriksa bidang `result.content`, itu akan kosong. Itu karena LLM telah memutuskan untuk memanggil alat, dan pesan keluaran memiliki petunjuk untuk itu. Yang terjadi di balik layar adalah bahwa LangChain memetakan format keluaran spesifik dari penyedia model ke format pemanggilan alat yang terunifikasi:

```python
print(result.tool_calls)
```

```
>> [{'name': 'google_search', 'args': {'query': 'age of Donald Trump'}, 'id': '6ab0de4b-f350-4743-a4c1-d6f6fcce9d34', 'type': 'tool_call'}]
```

Perlu diingat bahwa beberapa penyedia model mungkin mengembalikan konten yang tidak kosong bahkan dalam kasus pemanggilan alat (misalnya, mungkin ada jejak penalaran tentang mengapa model memutuskan untuk memanggil alat). Anda perlu melihat spesifikasi penyedia model untuk memahami bagaimana memperlakukan kasus seperti itu.

Seperti yang dapat kita lihat, LLM mengembalikan array kamus pemanggilan alat—masing-masing berisi pengidentifikasi unik, nama alat yang akan dipanggil, dan kamus dengan argumen yang akan diberikan ke alat ini. Mari kita lanjutkan ke langkah berikutnya dan panggil model lagi:

```python
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
tool_result = ToolMessage(content="Donald Trump ' Age 78 years June 14, 1946\n", tool_call_id=step1.tool_calls[0]["id"])
step2 = llm.invoke([
   HumanMessage(content=question), step1, tool_result], tools=[search_tool])
assert len(step2.tool_calls) == 0
print(step2.content)
```

```
>> Donald Trump is 78 years old.
```

`ToolMessage` adalah pesan khusus di LangChain yang memungkinkan Anda memberi umpan balik keluaran eksekusi alat kembali ke model. Bidang `content` dari pesan seperti itu berisi keluaran alat, dan bidang khusus `tool_call_id` memetakannya ke pemanggilan alat spesifik yang dihasilkan oleh model. Sekarang, kita dapat mengirim seluruh urutan (terdiri dari keluaran awal, langkah dengan pemanggilan alat, dan keluaran) kembali ke model sebagai daftar pesan.

Mungkin aneh untuk selalu meneruskan daftar alat ke LLM (karena, biasanya, daftar seperti itu tetap untuk alur kerja tertentu). Untuk alasan itu, `Runnables` LangChain menawarkan metode `bind` yang menghafal argumen dan menambahkannya ke setiap pemanggilan selanjutnya. Lihat kode berikut:

```python
llm_with_tools = llm.bind(tools=[search_tool])
llm_with_tools.invoke(question)
```

Ketika kita memanggil `llm.bind(tools=[search_tool])`, LangChain membuat objek baru (ditugaskan di sini ke `llm_with_tools`) yang secara otomatis menyertakan `[search_tool]` di setiap panggilan berikutnya ke salinan `llm` awal. Intinya, Anda tidak perlu lagi meneruskan argumen alat dengan setiap metode `invoke`. Jadi, memanggil kode sebelumnya sama dengan melakukan:

```python
llm.invoke(question, tools=[search_tool)
```

Ini karena bind telah "menghafal" daftar alat Anda untuk semua panggilan masa depan. Ini terutama fitur kenyamanan—ideal jika Anda ingin satu set alat tetap untuk panggilan berulang daripada menentukannya setiap kali. Sekarang mari kita lihat bagaimana kita dapat menggunakan pemanggilan alat lebih banyak lagi, dan meningkatkan penalaran LLM!

## ReACT

Seperti yang mungkin sudah Anda pikirkan, LLM dapat memanggil beberapa alat sebelum menghasilkan balasan akhir kepada pengguna (dan alat berikutnya yang akan dipanggil atau payload yang dikirim ke alat ini mungkin tergantung pada hasil dari panggilan alat sebelumnya). Ini diusulkan oleh pendekatan ReACT yang diperkenalkan pada tahun 2022 oleh peneliti dari Princeton University dan Google Research: _Reasoning and ACT_ ([https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)). Ideinya sederhana—kita harus memberi LLM akses ke alat sebagai cara untuk berinteraksi dengan lingkungan eksternal, dan biarkan LLM berjalan dalam lingkaran:

- **Reason**: Hasilkan keluaran teks dengan observasi tentang situasi saat ini dan rencana untuk menyelesaikan tugas.
- **Act**: Ambil tindakan berdasarkan penalaran di atas (berinteraksi dengan lingkungan dengan memanggil alat, atau merespons pengguna).

Telah ditunjukkan bahwa ReACT dapat membantu mengurangi tingkat halusinasi dibandingkan dengan prompt CoT, yang kita bahas di [Bab 3](Chapter_3.xhtml#_idTextAnchor049).

![Gambar 5.1: Pola ReACT](Images/B32363_05_01.png)

Mari kita bangun aplikasi ReACT sendiri. Pertama, mari buat alat pencarian dan kalkulator tiruan:

```python
import math
def mocked_google_search(query: str) -> str:
 print(f"CALLED GOOGLE_SEARCH with query={query}")
 return "Donald Trump is a president of USA and he's 78 years old"
def mocked_calculator(expression: str) -> float:
 print(f"CALLED CALCULATOR with expression={expression}")
 if "sqrt" in expression:
   return math.sqrt(78*132)
 return 78*132
```

Di bagian berikutnya, kita akan melihat bagaimana kita dapat membangun alat yang sebenarnya. Untuk saat ini, mari kita definisikan skema untuk alat kalkulator dan buat LLM sadar akan kedua alat yang dapat digunakannya. Kami juga akan menggunakan blok bangunan yang sudah kami kenal—`ChatPromptTemplate` dan `MessagesPlaceholder`—untuk menambahkan pesan sistem yang telah ditentukan ketika kami memanggil grafik kami:

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
calculator_tool = {
  "title": "calculator",
   "description": "Computes mathematical expressions",
  "type": "object",
  "properties": {
      "expression": {
          "description": "A mathematical expression to be evaluated by a calculator",
          "title": "expression",
          "type": "string"},
  },
  "required": ["expression"]
}
prompt = ChatPromptTemplate.from_messages([
   ("system", "Always use a calculator for mathematical computations, and use Google Search for information about fresh events and news."),
   MessagesPlaceholder(variable_name="messages"),
])
llm_with_tools = llm.bind(tools=[search_tool, calculator_tool]).bind(prompt=prompt)
```

Sekarang kita memiliki LLM yang dapat memanggil alat, mari buat simpul yang kita butuhkan. Kita membutuhkan satu fungsi yang memanggil LLM, fungsi lain yang memanggil alat dan mengembalikan hasil pemanggilan alat (dengan menambahkan `ToolMessages` ke daftar pesan di status), dan fungsi yang akan menentukan apakah pengatur harus terus memanggil alat atau apakah dapat mengembalikan hasil kepada pengguna:

```python
from typing import TypedDict
from langgraph.graph import MessagesState, StateGraph, START, END
def invoke_llm(state: MessagesState):
   return {"messages": [llm_with_tools.invoke(state["messages"])]}
def call_tools(state: MessagesState):
   last_message = state["messages"][-1]
   tool_calls = last_message.tool_calls
   new_messages = []
   for tool_call in tool_calls:
     if tool_call["name"] == "google_search":
       tool_result = mocked_google_search(**tool_call["args"])
       new_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
     elif tool_call["name"] == "calculator":
       tool_result = mocked_calculator(**tool_call["args"])
       new_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
     else:
       raise ValueError(f"Tool {tool_call['name']} is not defined!")
   return {"messages": new_messages}
def should_run_tools(state: MessagesState):
   last_message = state["messages"][-1]
   if last_message.tool_calls:
     return "call_tools"
   return END
```

Sekarang mari kita satukan semuanya dalam alur kerja LangGraph:

```python
builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("call_tools", call_tools)
builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", should_run_tools)
builder.add_edge("call_tools", "invoke_llm")
graph = builder.compile()
question = "What is a square root of the current US president's age multiplied by 132?"
result = graph.invoke({"messages": [HumanMessage(content=question)]})
print(result["messages"][-1].content)
```

```
>> CALLED GOOGLE_SEARCH with query=age of Donald Trump
CALLED CALCULATOR with expression=78 * 132
CALLED CALCULATOR with expression=sqrt(10296)
The square root of 78 multiplied by 132 (which is 10296) is approximately 101.47.
```

Ini menunjukkan bagaimana LLM melakukan beberapa panggilan untuk menangani pertanyaan kompleks—pertama, ke `Google Search` dan kemudian dua panggilan ke `Calculator`—dan setiap kali, ia menggunakan informasi yang sebelumnya diterima untuk menyesuaikan tindakannya. Inilah pola ReACT dalam aksi.

Dengan itu, kita telah mempelajari bagaimana pola ReACT bekerja secara detail dengan membangunnya sendiri. Kabar baiknya adalah bahwa LangGraph menawarkan implementasi bawaan dari pola ReACT, sehingga Anda tidak perlu mengimplementasikannya sendiri:

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
  llm=llm,
  tools=[search_tool, calculator_tool],
  prompt=system_prompt)
```

Di [Bab 6](Chapter_6.xhtml#_idTextAnchor132), kita akan melihat beberapa penyesuaian tambahan yang dapat Anda gunakan dengan fungsi `create_react_agent`.

# Mendefinisikan alat

Sejauh ini, kita telah mendefinisikan alat sebagai skema OpenAPI. Tetapi untuk menjalankan alur kerja dari ujung ke ujung, LangGraph harus dapat memanggil alat itu sendiri selama eksekusi. Oleh karena itu, di bagian ini, mari kita bahas bagaimana kita mendefinisikan alat sebagai fungsi atau callable Python.

Alat LangChain memiliki tiga komponen penting:

- `Nama`: Pengidentifikasi unik untuk alat
- `Deskripsi`: Teks yang membantu LLM memahami kapan dan bagaimana menggunakan alat
- `Skema payload`: Definisi terstruktur dari input yang diterima alat

Ini memungkinkan LLM untuk memutuskan kapan dan bagaimana memanggil alat. Perbedaan penting lainnya dari alat LangChain adalah bahwa alat dapat dieksekusi oleh pengatur, seperti LangGraph. Antarmuka dasar untuk alat adalah `BaseTool`, yang mewarisi dari `RunnableSerializable` itu sendiri. Itu berarti alat dapat dipanggil atau dikelompokkan sebagai `Runnable` apa pun, atau diserialisasi atau dideserialisasi sebagai `Serializable` apa pun.

## Alat LangChain bawaan

LangChain memiliki banyak alat yang sudah tersedia di berbagai kategori. Karena alat sering disediakan oleh vendor pihak ketiga, beberapa alat memerlukan kunci API berbayar, beberapa sepenuhnya gratis, dan beberapa memiliki tingkat gratis. Beberapa alat dikelompokkan bersama dalam toolkit—kumpulan alat yang seharusnya digunakan bersama saat mengerjakan tugas tertentu. Mari kita lihat beberapa contoh penggunaan alat.

Alat memberi LLM akses ke mesin pencari, seperti Bing, DuckDuckGo, Google, dan Tavily. Mari kita lihat `DuckDuckGoSearchRun` karena mesin pencari ini tidak memerlukan pendaftaran tambahan dan kunci API.

Silakan lihat [Bab 2](Chapter_2.xhtml#_idTextAnchor025) untuk instruksi penyiapan. Jika Anda memiliki pertanyaan atau mengalami masalah saat menjalankan kode, silakan buat isu di GitHub atau bergabung dalam diskusi di Discord di [https://packt.link/lang](https://packt.link/lang).

Seperti alat apa pun, alat ini memiliki nama, deskripsi, dan skema untuk argumen input:

```python
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
print(f"Tool's name = {search.name}")
print(f"Tool's name = {search.description}")
print(f"Tool's arg schema = f{search.args_schema}")
```

```
>> Tool's name = fduckduckgo_search
Tool's name = fA wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
Tool's arg schema = class 'langchain_community.tools.ddg_search.tool.DDGInput'
```

Skema argumen, `arg_schema`, adalah model Pydantic dan kita akan melihat mengapa itu berguna nanti di bab ini. Kita dapat mengeksplorasi bidangnya secara terprogram atau dengan pergi ke halaman dokumentasi—ini hanya mengharapkan satu bidang input, kueri:

```python
from langchain_community.tools.ddg_search.tool import DDGInput
print(DDGInput.__fields__)
```

```
>> {'query': FieldInfo(annotation=str, required=True, description='search query to look up')}
```

Sekarang kita dapat memanggil alat ini dan mendapatkan keluaran string kembali (hasil dari mesin pencari):

```python
query = "What is the weather in Munich like tomorrow?"
search_input = DDGInput(query=query)
result = search.invoke(search_input.dict())
print(result)
```

Kami juga dapat memanggil LLM dengan alat, dan pastikan bahwa LLM memanggil alat pencarian dan tidak menjawab langsung:

```python
result = llm.invoke(query, tools=[search])
print(result.tool_calls[0])
```

```
>> {'name': 'duckduckgo_search', 'args': {'query': 'weather in Munich tomorrow'}, 'id': '222dc19c-956f-4264-bf0f-632655a6717d', 'type': 'tool_call'}
```

Alat kami sekarang adalah callable yang dapat dipanggil secara terprogram oleh LangGraph. Mari kita satukan semuanya dan buat agen pertama kami. Ketika kita streaming grafik kami, kami mendapatkan pembaruan ke status. Dalam kasus kami, ini hanya pesan:

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model=llm, tools=[search])
```

![Gambar 5.2: Alur kerja ReACT bawaan di LangGraph](Images/B32363_05_02.png)

Itu persis seperti yang kita lihat sebelumnya juga—LLM memanggil alat sampai memutuskan untuk berhenti dan mengembalikan jawaban kepada pengguna. Mari kita uji!

Ketika kita streaming LangGraph, kita mendapatkan peristiwa baru yang merupakan pembaruan status grafik. Kami tertarik pada bidang `message` dari status. Mari cetak pesan baru yang ditambahkan:

```python
for event in agent.stream({"messages": [("user", query)]}):
 update = event.get("agent", event.get("tools", {}))
 for message in update.get("messages", []):
    message.pretty_print()
```

```
>> ================================ Ai Message ==================================
Tool Calls:
  duckduckgo_search (a01a4012-bfc0-4eae-9c81-f11fd3ecb52c)
 Call ID: a01a4012-bfc0-4eae-9c81-f11fd3ecb52c
  Args:
    query: weather in Munich tomorrow
================================= Tool Message =================================
Name: duckduckgo_search
The temperature in Munich tomorrow in the early morning is 4 ° C… <TRUNCATED>
================================== Ai Message ==================================
The weather in Munich tomorrow will be 5°C with a 0% chance of rain in the morning.  The wind will blow at 11 km/h.  Later in the day, the high will be 53°F (approximately 12°C).  It will be clear in the early morning.
```

Agen kami diwakili oleh daftar pesan karena ini adalah input dan output yang diharapkan LLM. Kita akan melihat pola itu lagi ketika kita menyelami lebih dalam arsitektur agen dan membahasnya di bab berikutnya. Untuk saat ini, mari sebutkan secara singkat jenis alat lain yang sudah tersedia di LangChain:

- **Alat yang meningkatkan pengetahuan LLM selain menggunakan mesin pencari**:
  - Penelitian akademik: arXiv dan PubMed
  - Basis pengetahuan: Wikipedia dan Wikidata
  - Data keuangan: Alpha Vantage, Polygon, dan Yahoo Finance
  - Cuaca: OpenWeatherMap
  - Komputasi: Wolfram Alpha
- **Alat yang meningkatkan produktivitas Anda**: Anda dapat berinteraksi dengan Gmail, Slack, Office 365, Google Calendar, Jira, Github, dll. Misalnya, `GmailToolkit` memberi Anda akses ke alat `GmailCreateDraft`, `GmailSendMessage`, `GmailSearch`, `GmailGetMessage`, dan `GmailGetThread` yang memungkinkan Anda mencari, mengambil, membuat, dan mengirim pesan dengan akun Gmail Anda. Seperti yang Anda lihat, tidak hanya Anda dapat memberi LLM konteks tambahan tentang pengguna tetapi, dengan beberapa alat ini, LLM dapat mengambil tindakan yang benar-benar mempengaruhi lingkungan luar, seperti membuat pull request di GitHub atau mengirim pesan di Slack!
- **Alat yang memberi LLM akses ke penerjemah kode**: Alat ini memberi LLM akses ke penerjemah kode dengan meluncurkan wadah terisolasi dari jarak jauh dan memberi LLM akses ke wadah ini. Alat ini memerlukan kunci API dari vendor yang menyediakan sandbox. LLM sangat pandai dalam coding, dan ini adalah pola yang banyak digunakan untuk meminta LLM menyelesaikan beberapa tugas kompleks dengan menulis kode yang menyelesaikannya alih-alih memintanya untuk menghasilkan token yang mewakili solusi tugas. Tentu saja, Anda harus mengeksekusi kode yang dihasilkan LLM dengan hati-hati, dan itulah mengapa sandbox terisolasi memainkan peran besar. Beberapa contoh adalah:
  - Eksekusi kode: Python REPL dan Bash
  - Layanan cloud: AWS Lambda
  - Alat API: GraphQL dan Requests
  - Operasi file: Sistem File
- **Alat yang memberi LLM akses ke basis data dengan menulis dan mengeksekusi kode SQL**: Misalnya, `SQLDatabase` menyertakan alat untuk mendapatkan informasi tentang basis data dan objeknya dan mengeksekusi kueri SQL. Anda juga dapat mengakses Google Drive dengan `GoogleDriveLoader` atau melakukan operasi dengan alat sistem file biasa dari `FileManagementToolkit`.
- **Alat lain**: Ini terdiri dari alat yang mengintegrasikan sistem pihak ketiga dan memungkinkan LLM mengumpulkan informasi tambahan atau bertindak. Ada juga alat yang dapat mengintegrasikan pengambilan data dari Google Maps, NASA, dan platform serta organisasi lain.
- **Alat untuk menggunakan sistem AI lain atau otomatisasi**:
  - Pembuatan gambar: DALL-E dan Imagen
  - Sintesis ucapan: Google Cloud TTS dan Eleven Labs
  - Akses model: Hugging Face Hub
  - Otomatisasi alur kerja: Zapier dan IFTTT

Setiap sistem eksternal dengan API dapat dibungkus sebagai alat jika meningkatkan LLM seperti ini:

- Menyediakan pengetahuan domain yang relevan kepada pengguna atau alur kerja
- Memungkinkan LLM mengambil tindakan atas nama pengguna

Saat mengintegrasikan alat seperti itu dengan LangChain, pertimbangkan aspek kunci berikut:

- **Autentikasi**: Akses aman ke sistem eksternal
- **Skema payload**: Tentukan struktur data yang tepat untuk input/output
- **Penanganan kesalahan**: Rencanakan untuk kegagalan dan kasus tepi
- **Pertimbangan keamanan**: Misalnya, saat mengembangkan agen SQL-ke-teks, batasi akses ke operasi baca-saja untuk mencegah modifikasi yang tidak diinginkan

Oleh karena itu, toolkit penting adalah `RequestsToolkit`, yang memungkinkan seseorang dengan mudah membungkus API HTTP apa pun:

```python
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
toolkit = RequestsToolkit(
   requests_wrapper=TextRequestsWrapper(headers={}),
   allow_dangerous_requests=True,
)
for tool in toolkit.get_tools():
 print(tool.name)
```

```
>> requests_get
requests_post
requests_patch
requests_put
requests_delete
```

Mari kita ambil API mata uang sumber terbuka gratis ([https://frankfurter.dev/](https://frankfurter.dev/)). Ini adalah API gratis acak yang kami ambil dari Internet hanya untuk tujuan ilustrasi, hanya untuk menunjukkan kepada Anda bagaimana Anda dapat membungkus API yang ada sebagai alat. Pertama, kita perlu menyusun spesifikasi API berdasarkan format OpenAPI. Kami memotong spesifikasi tetapi Anda dapat menemukan versi lengkapnya di GitHub kami:

```python
api_spec = """
openapi: 3.0.0
info:
 title: Frankfurter Currency Exchange API
 version: v1
 description: API for retrieving currency exchange rates. Pay attention to the base currency and change it if needed.
servers:
 - url: https://api.frankfurter.dev/v1
paths:
 /v1/latest:
   get:
     summary: Get the latest exchange rates.
     parameters:
       - in: query
         name: symbols
         schema:
           type: string
         description: Comma-separated list of currency symbols to retrieve rates for. Example: CHF,GBP
       - in: query
         name: base
         schema:
           type: string
         description: The base currency for the exchange rates. If not provided, EUR is used as a base currency. Example: USD
   /v1/{date}:
   ...
"""
```

Sekarang mari kita bangun dan jalankan agen ReACT kami; kita akan melihat bahwa LLM dapat mengkueri API pihak ketiga dan memberikan jawaban terkini tentang kurs mata uang:

```python
system_message = (
 "You're given the API spec:\n{api_spec}\n"
 "Use the API to answer users' queries if possible. "
)
agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=system_message.format(api_spec=api_spec))
query = "What is the swiss franc to US dollar exchange rate?"
events = agent.stream(
   {"messages": [("user", query)]},
   stream_mode="values",
)
for event in events:
   event["messages"][-1].pretty_print()
```

```
>> ============================== Human Message =================================
What is the swiss franc to US dollar exchange rate?
================================== Ai Message ==================================
Tool Calls:
  requests_get (541a9197-888d-4ffe-a354-c726804ad7ff)
 Call ID: 541a9197-888d-4ffe-a354-c726804ad7ff
  Args:
    url: https://api.frankfurter.dev/v1/latest?symbols=CHF&base=USD
================================= Tool Message =================================
Name: requests_get
{"amount":1.0,"base":"USD","date":"2025-01-31","rates":{"CHF":0.90917}}
================================== Ai Message ==================================
The Swiss franc to US dollar exchange rate is 0.90917.
```

Perhatikan bahwa, kali ini, kami menggunakan opsi `stream_mode="values"`, dan dalam opsi ini, setiap kali, kami mendapatkan status penuh saat ini dari grafik.

<div class="note">
Ada lebih dari 50 alat yang sudah tersedia. Anda dapat menemukan daftar lengkap di halaman dokumentasi: [https://python.langchain.com/docs/integrations/tools/](https://python.langchain.com/docs/integrations/tools/). 
</div>

## Alat kustom

Kami melihat berbagai alat bawaan yang ditawarkan oleh LangGraph. Sekarang saatnya membahas bagaimana Anda dapat membuat alat kustom sendiri, selain contoh yang kami lihat ketika kami membungkus API pihak ketiga dengan `RequestsToolkit` dengan memberikan spesifikasi API. Mari kita mulai!

### Membungkus fungsi Python sebagai alat

Setiap fungsi `Python` (atau callable) dapat dibungkus sebagai alat. Seperti yang kita ingat, alat di LangChain harus memiliki nama, deskripsi, dan skema argumen. Mari kita bangun kalkulator kita sendiri berdasarkan pustaka Python `numexr`—pengevaluasi ekspresi numerik cepat berbasis NumPy ([https://github.com/pydata/numexpr](https://github.com/pydata/numexpr)). Kami akan menggunakan dekorator khusus `@tool` yang akan membungkus fungsi kami sebagai alat:

```python
import math
from langchain_core.tools import tool
import numexpr as ne
@tool
def calculator(expression: str) -> str:
   """Calculates a single mathematical expression, incl. complex numbers.

    Always add * to operations, examples:
      73i -> 73*i
      7pi**2 -> 7*pi**2
   """
   math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
   result = ne.evaluate(expression.strip(), local_dict=math_constants)
   return str(result)
```

Mari kita jelajahi objek kalkulator yang kita miliki! Perhatikan bahwa LangChain mewarisi nama, deskripsi, dan skema args dari docstring dan petunjuk tipe secara otomatis. Harap dicatat bahwa kami menggunakan teknik few-shot (dibahas di [Bab 3](Chapter_3.xhtml#_idTextAnchor049)) untuk mengajarkan LLM cara menyiapkan payload untuk alat kami dengan menambahkan dua contoh dalam docstring:

```python
from langchain_core.tools import BaseTool
assert isinstance(calculator, BaseTool)
print(f"Tool schema: {calculator.args_schema.model_json_schema()}")
```

```
>> Tool schema: {'description': 'Calculates a single mathematical expression, incl. complex numbers.\n\nAlways add * to operations, examples:\n  73i -> 73*i\n  7pi**2 -> 7*pi**2', 'properties': {'expression': {'title': 'Expression', 'type': 'string'}}, 'required': ['expression'], 'title': 'calculator', 'type': 'object'}
```

Mari kita coba alat baru kami untuk mengevaluasi ekspresi dengan bilangan kompleks, yang memperluas bilangan real dengan unit imajiner khusus `i` yang memiliki properti `i**2=-1`:

```python
query = "How much is 2+3i squared?"
agent = create_react_agent(llm, [calculator])
for event in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
   event["messages"][-1].pretty_print()
```

```
>> ===============================Human Message =================================
How much is 2+3i squared?
================================== Ai Message ==================================
Tool Calls:
  calculator (9b06de35-a31c-41f3-a702-6e20698bf21b)
 Call ID: 9b06de35-a31c-41f3-a702-6e20698bf21b
  Args:
    expression: (2+3*i)**2
================================= Tool Message =================================
Name: calculator
(-5+12j)
================================== Ai Message ==================================
(2+3i)² = -5+12i.
```

Dengan hanya beberapa baris kode, kami telah berhasil memperluas kemampuan LLM kami untuk bekerja dengan bilangan kompleks. Sekarang kita dapat menyatukan contoh yang kita mulai:

```python
question = "What is a square root of the current US president's age multiplied by 132?"
system_hint = "Think step-by-step. Always use search to get the fresh information about events or public facts that can change over time."
agent = create_react_agent(
   llm, [calculator, search],
   state_modifier=system_hint)
for event in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
   event["messages"][-1].pretty_print()
print(event["messages"][-1].content)
```

```
>> The square root of Donald Trump's age multiplied by 132 is approximately 101.47.
```

Kami belum memberikan keluaran lengkap di sini dalam buku (Anda dapat menemukannya di GitHub kami), tetapi jika Anda menjalankan cuplikan ini, Anda harus melihat bahwa LLM dapat mengkueri alat langkah demi langkah:

1. Ini memanggil mesin pencari dengan kueri `"current US president"`.
2. Kemudian, lagi memanggil mesin pencari dengan kueri `"donald trump age"`.
3. Sebagai langkah terakhir, LLM memanggil alat kalkulator dengan ekspresi `"sqrt(78*132)"`.
4. Akhirnya, ia mengembalikan jawaban yang benar kepada pengguna.

Di setiap langkah, LLM bernalar berdasarkan informasi yang sebelumnya dikumpulkan dan kemudian bertindak dengan alat yang sesuai—itulah inti dari pendekatan ReACT.

### Membuat alat dari Runnable

Terkadang, LangChain mungkin tidak dapat menurunkan deskripsi atau skema args yang tepat dari suatu fungsi, atau kita mungkin menggunakan callable kompleks yang sulit dibungkus dengan dekorator. Misalnya, kita dapat menggunakan rantai LangChain lain atau grafik LangGraph sebagai alat. Kita dapat membuat alat dari `Runnable` apa pun dengan secara eksplisit menentukan semua deskripsi yang diperlukan. Mari buat alat kalkulator dari fungsi dengan cara alternatif, dan kami akan menyetel perilaku coba ulang (dalam kasus kami, kami akan mencoba tiga kali dan menambakang backoff eksponensial antara upaya berturut-turut):

<div class="note">
Harap dicatat bahwa kami menggunakan fungsi yang sama seperti di atas tetapi kami menghapus dekorator `@tool`.
</div>

```python
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.tools import tool, convert_runnable_to_tool
def calculator(expression: str) -> str:
   math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
   result = ne.evaluate(expression.strip(), local_dict=math_constants)
   return str(result)
calculator_with_retry = RunnableLambda(calculator).with_retry(
   wait_exponential_jitter=True,
   stop_after_attempt=3,
)
calculator_tool = convert_runnable_to_tool(
   calculator_with_retry,
   name="calculator",
   description=(
       "Calculates a single mathematical expression, incl. complex numbers."
       "'\nAlways add * to operations, examples:\n73i -> 73*i\n"
       "7pi**2 -> 7*pi**2"
   ),
   arg_types={"expression": "str"},
)
```

Perhatikan bahwa kami mendefinisikan fungsi kami dengan cara yang mirip dengan bagaimana kami mendefinisikan simpul LangGraph—ini mengambil status (yang sekarang adalah model Pydantic) dan konfigurasi. Kemudian, kami membungkus fungsi ini sebagai `RunnableLambda` dan menambahkan coba ulang. Ini mungkin berguna jika kami ingin menjaga fungsi Python kami sebagai fungsi tanpa membungkusnya dengan dekorator, atau jika kami ingin membungkus API eksternal (oleh karena itu, deskripsi dan skema argumen tidak dapat diwarisi secara otomatis dari docstring). Kami dapat menggunakan Runnable apa pun (misalnya, rantai atau grafik) untuk membuat alat, dan itu memungkinkan kami membangun sistem multi-agen karena sekarang satu alur kerja berbasis LLM dapat memanggil alur kerja berbasis LLM lainnya. Mari kita ubah Runnable kami menjadi alat:

```python
calculator_tool = convert_runnable_to_tool(
   calculator_with_retry,
   name="calculator",
   description=(
       "Calculates a single mathematical expression, incl. complex numbers."
       "'\nAlways add * to operations, examples:\n73i -> 73*i\n"
       "7pi**2 -> 7*pi**2"
   ),
   arg_types={"expression": "str"},
)
```

Mari kita uji fungsi `calculator` baru kami dengan LLM:

```python
llm.invoke("How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0]
```

```
>> {'name': 'calculator',
 'args': {'__arg1': '(2+3*i)**2'},
 'id': '46c7e71c-4092-4299-8749-1b24a010d6d6',
 'type': 'tool_call'}
```

Seperti yang dapat Anda catat, LangChain tidak mewarisi skema `args` sepenuhnya; itulah sebabnya ia membuat nama buatan untuk argumen seperti `__arg1`. Mari ubah alat kami untuk menerima model Pydantic sebagai gantinya, dengan cara yang mirip dengan bagaimana kami mendefinisikan simpul LangGraph:

```python
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
class CalculatorArgs(BaseModel):
   expression: str = Field(description="Mathematical expression to be evaluated")
def calculator(state: CalculatorArgs, config: RunnableConfig) -> str:
   expression = state["expression"]
   math_constants = config["configurable"].get("math_constants", {})
   result = ne.evaluate(expression.strip(), local_dict=math_constants)
   return str(result)
```

Sekarang skema lengkap adalah yang tepat:

```python
assert isinstance(calculator_tool, BaseTool)
print(f"Tool name: {calculator_tool.name}")
print(f"Tool description: {calculator_tool.description}")
print(f"Args schema: {calculator_tool.args_schema.model_json_schema()}")
```

```
>> Tool name: calculator
Tool description: Calculates a single mathematical expression, incl. complex numbers.'
Always add * to operations, examples:
73i -> 73*i
7pi**2 -> 7*pi**2
Args schema: {'properties': {'expression': {'title': 'Expression', 'type': 'string'}}, 'required': ['expression'], 'title': 'calculator', 'type': 'object'}
```

Mari kita uji bersama dengan LLM:

```python
tool_call = llm.invoke("How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0]
print(tool_call)
```

```
>> {'name': 'calculator', 'args': {'expression': '(2+3*i)**2'}, 'id': 'f8be9cbc-4bdc-4107-8cfb-fd84f5030299', 'type': 'tool_call'}
```

Kami dapat memanggil alat kalkulator kami dan meneruskannya ke konfigurasi LangGraph dalam runtime:

```python
math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
config = {"configurable": {"math_constants": math_constants}}
calculator_tool.invoke(tool_call["args"], config=config)
```

```
>> (-5+12j)
```

Dengan itu, kita telah belajar bagaimana kita dapat dengan mudah mengubah Runnable apa pun menjadi alat dengan memberikan detail tambahan ke LangChain untuk memastikan LLM dapat menangani alat ini dengan benar.

### Subkelas StructuredTool atau BaseTool

Metode lain untuk mendefinisikan alat adalah dengan membuat alat kustom dengan membuat subkelas dari kelas `BaseTool`. Seperti pendekatan lain, Anda harus menentukan nama alat, deskripsi, dan skema argumen. Anda juga perlu mengimplementasikan satu atau dua metode abstrak: `_run` untuk eksekusi sinkron dan, jika perlu, `_arun` untuk perilaku asinkron (jika berbeda dari hanya membungkus versi sinkron). Opsi ini sangat berguna ketika alat Anda perlu menyimpan status (misalnya, untuk mempertahankan klien koneksi jangka panjang) atau ketika logikanya terlalu kompleks untuk diimplementasikan sebagai fungsi tunggal atau `Runnable`.

Jika Anda ingin fleksibilitas lebih daripada yang diberikan dekorator `@tool` tetapi tidak ingin mengimplementasikan kelas Anda sendiri, ada pendekatan perantara. Anda juga dapat menggunakan metode kelas `StructuredTool.from_function`, yang memungkinkan Anda untuk secara eksplisit menentukan parameter meta alat seperti deskripsi atau `args_schema` hanya dengan beberapa baris kode:

```python
from langchain_core.tools import StructuredTool
calculator_tool = StructuredTool.from_function(
   name="calculator",
   description=(
       "Calculates a single mathematical expression, incl. complex numbers."),
   func=calculator,
   args_schema=CalculatorArgs
)
tool_call = llm.invoke(
  "How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0]
```

Satu catatan terakhir tentang implementasi sinkron dan asinkron diperlukan pada titik ini. Jika fungsi mendasar selain alat Anda adalah fungsi sinkron, LangChain akan membungkusnya untuk implementasi asinkron alat dengan meluncurkannya di utas terpisah. Dalam banyak kasus, ini tidak masalah, tetapi jika Anda peduli dengan overhead tambahan dari pembuatan utas terpisah, Anda memiliki dua opsi—baik membuat subkelas dari `BaseClass` dan mengganti implementasi async, atau membuat implementasi async terpisah dari fungsi Anda dan meneruskannya ke `StructruredTool.from_function` sebagai argumen `coroutine`. Anda juga dapat memberikan hanya implementasi async, tetapi kemudian Anda tidak akan dapat memanggil alur kerja Anda secara sinkron.

Sebagai kesimpulan, mari kita lihat lagi tiga opsi yang kita miliki untuk membuat alat LangChain, dan kapan menggunakan masing-masingnya.

| **Metode untuk membuat alat**                      | **Kapan digunakan**                                                                                                                                                                                                    |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dekorator @tool                                    | Anda memiliki fungsi dengan docstring yang jelas dan fungsi ini tidak digunakan di mana pun dalam kode Anda                                                                                                            |
| convert_runnable_to_tool                           | Anda memiliki Runnable yang ada, atau Anda memerlukan kendali lebih terperinci tentang bagaimana argumen atau deskripsi alat diteruskan ke LLM (Anda membungkus fungsi yang ada dengan RunnableLambda dalam kasus itu) |
| Membuat subkelas dari StructuredTool atau BaseTool | Anda memerlukan kendali penuh atas deskripsi dan logika alat (misalnya, Anda ingin menangani permintaan sinkron dan async secara berbeda)                                                                              |

Tabel 5.1: Opsi untuk membuat alat LangChain

Ketika LLM menghasilkan payload dan memanggil alat, ia mungkin berhalusinasi atau membuat kesalahan lain. Oleh karena itu, kita perlu memikirkan penanganan kesalahan dengan hati-hati.

## Penanganan kesalahan

Kami sudah membahas penanganan kesalahan di [Bab 3](Chapter_3.xhtml#_idTextAnchor049), tetapi ini menjadi lebih penting ketika Anda meningkatkan LLM dengan alat; Anda membutuhkan pencatatan, bekerja dengan pengecualian, dan sebagainya bahkan lebih. Pertimbangan tambahan adalah memikirkan apakah Anda ingin alur kerja Anda melanjutkan dan mencoba memulihkan sendiri jika salah satu alat Anda gagal. LangChain memiliki `ToolException` khusus yang memungkinkan alur kerja melanjutkan eksekusinya dengan menangani pengecualian.

`BaseTool` memiliki dua bendera khusus: `handle_tool_error` dan `handle_validation_error`. Tentu saja, karena `StructuredTool` mewarisi dari `BaseTool`, Anda dapat meneruskan bendera ini ke metode kelas `StructuredTool.from_function`. Jika bendera ini disetel, LangChain akan membuat string untuk dikembalikan sebagai hasil eksekusi alat jika terjadi `ToolException` atau `ValidationException` Pydantic (saat memvalidasi payload input).

Untuk memahami apa yang terjadi, mari kita lihat kode sumber LangChain untuk fungsi `_handle_tool_error`:

```python
def _handle_tool_error(
  e: ToolException,
  *,
  flag: Optional[Union[Literal[True], str, Callable[[ToolException], str]]],
) -> str:
    if isinstance(flag, bool):
        content = e.args[0] if e.args else "Tool execution error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got an unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)  # noqa: TRY004
    return content
```

Seperti yang dapat kita lihat, kita dapat mengatur bendera ini ke Boolean, string, atau callable (yang mengubah `ToolException` menjadi string). Berdasarkan ini, LangChain akan mencoba menangani `ToolException` dan meneruskan string ke tahap berikutnya sebagai gantinya. Kita dapat menggabungkan umpan balik ini ke dalam alur kerja kami dan menambahkan loop pemulihan otomatis.

Mari kita lihat contoh. Kami menyesuaikan fungsi `calculator` kami dengan menghapus substitusi `i->j` (substitusi dari unit imajiner dalam matematika ke unit imajiner dalam Python), dan kami juga membuat `StructuredTool` mewarisi deskripsi dan `arg_schema` dari docstring secara otomatis:

```python
from langchain_core.tools import StructuredTool
def calculator(expression: str) -> str:
   """Calculates a single mathematical expression, incl. complex numbers."""
   return str(ne.evaluate(expression.strip(), local_dict={}))
calculator_tool = StructuredTool.from_function(
   func=calculator,
   handle_tool_error=True
)
agent = create_react_agent(
   llm, [calculator_tool])
for event in agent.stream({"messages": [("user", "How much is (2+3i)^2")]}, stream_mode="values"):
   event["messages"][-1].pretty_print()
```

```
>> ============================== Human Message =================================
How much is (2+3i)^2
================================== Ai Message ==================================
Tool Calls:
  calculator (8bfd3661-d2e1-4b8d-84f4-0be4892d517b)
 Call ID: 8bfd3661-d2e1-4b8d-84f4-0be4892d517b
  Args:
    expression: (2+3i)^2
================================= Tool Message =================================
Name: calculator
Error: SyntaxError('invalid decimal literal', ('<expr>', 1, 4, '(2+3i)^2', 1, 4))
 Please fix your mistakes.
================================== Ai Message ==================================
(2+3i)^2 is equal to -5 + 12i.  I tried to use the calculator tool, but it returned an error. I will calculate it manually for you.
(2+3i)^2 = (2+3i)*(2+3i) = 2*2 + 2*3i + 3i*2 + 3i*3i = 4 + 6i + 6i - 9 = -5 + 12i
```

Seperti yang dapat kita lihat, sekarang eksekusi kalkulator kami gagal, tetapi karena deskripsi kesalahan tidak cukup jelas, LLM memutuskan untuk merespons sendiri tanpa menggunakan alat. Tergantung pada kasus penggunaan Anda, Anda mungkin ingin menyesuaikan perilaku; misalnya, berikan kesalahan yang lebih bermakna dari alat, paksa alur kerja untuk mencoba menyesuaikan payload untuk alat, dll.

LangGraph juga menawarkan `ValidationNode` bawaan yang mengambil pesan terakhir (dengan memeriksa kunci `messages` dalam status grafik) dan memeriksa apakah itu memiliki panggilan alat. Jika itu terjadi, LangGraph memvalidasi skema panggilan alat, dan jika tidak mengikuti skema yang diharapkan, itu memunculkan `ToolMessage` dengan kesalahan validasi (dan perintah default untuk memperbaikinya). Anda dapat menambahkan tepi bersyarat yang mengembalikan ke LLM dan kemudian LLM akan menghasilkan ulang panggilan alat, mirip dengan pola yang kita bahas di [Bab 3](Chapter_3.xhtml#_idTextAnchor049).

Sekarang setelah kita belajar apa itu alat, bagaimana membuatnya, dan bagaimana menggunakan alat LangChain bawaan, saatnya untuk melihat instruksi tambahan yang dapat Anda berikan kepada LLM tentang cara menggunakan alat.

# Kemampuan pemanggilan alat lanjutan

Banyak LLM menawarkan Anda beberapa opsi konfigurasi tambahan pada pemanggilan alat. Pertama, beberapa model mendukung pemanggilan fungsi paralel—khususnya, LLM dapat memanggil beberapa alat sekaligus. LangChain secara asli mendukung ini karena bidang `tool_calls` dari `AIMessage` adalah daftar. Ketika Anda mengembalikan objek `ToolMessage` sebagai hasil panggilan fungsi, Anda harus mencocokkan dengan hati-hati bidang `tool_call_id` dari `ToolMessage` ke payload yang dihasilkan. Penyelarasan ini diperlukan agar LangChain dan LLM yang mendasarinya dapat mencocokkannya bersama-sama saat melakukan giliran berikutnya.

Kemampuan lanjutan lainnya adalah memaksa LLM untuk memanggil alat, atau bahkan memanggil alat tertentu. Secara umum, LLM memutuskan apakah harus memanggil alat, dan jika harus, alat mana yang akan dipanggil dari daftar alat yang disediakan. Biasanya, ini ditangani oleh argumen `tool_choice` dan/atau `tool_config` yang diteruskan ke metode `invoke`, tetapi implementasi tergantung pada penyedia model. Anthropic, Google, OpenAI, dan penyedia utama lainnya memiliki API yang sedikit berbeda, dan meskipun LangChain mencoba menyatukan argumen, dalam kasus seperti itu, Anda harus memeriksa detailnya oleh penyedia model.

Biasanya, opsi berikut tersedia:

- `"auto"`: LLM dapat merespons atau memanggil satu atau banyak alat.
- `"any"`: LLM dipaksa untuk merespons dengan memanggil satu atau banyak alat.
- `"tool"` atau `"any"` dengan daftar alat yang disediakan: LLM dipaksa untuk merespons dengan memanggil alat dari daftar yang dibatasi.
- `"None"`: LLM dipaksa untuk merespons tanpa memanggil alat.

Hal penting lain yang perlu diingat adalah bahwa skema mungkin menjadi cukup kompleks—yaitu, mereka mungkin memiliki bidang yang dapat dinull, bidang bersarang, menyertakan enum, atau merujuk ke skema lain. Tergantung pada penyedia model, beberapa definisi mungkin tidak didukung (dan Anda akan melihat peringatan atau kesalahan kompilasi). Meskipun LangChain bertujuan untuk membuat pergantian di antara vendor menjadi mulus, untuk beberapa alur kerja yang kompleks, ini mungkin tidak terjadi, jadi perhatikan peringatan di log kesalahan. Terkadang, kompilasi skema yang disediakan ke skema yang didukung oleh penyedia model dilakukan dengan upaya terbaik—misalnya, bidang dengan tipe `Union[str, int]` dikompilasi ke tipe `str` jika LLM yang mendasarinya tidak mendukung tipe `Union` dengan pemanggilan alat. Anda akan mendapatkan peringatan, tetapi mengabaikan peringatan seperti itu selama migrasi dapat mengubah perilaku aplikasi Anda secara tak terduga.

Sebagai catatan akhir, perlu disebutkan bahwa beberapa penyedia (misalnya, OpenAI atau Google) menawarkan alat kustom, seperti penerjemah kode atau pencarian Google, yang dapat dipanggil oleh model itu sendiri, dan model akan menggunakan keluaran alat untuk menyiapkan generasi akhir. Anda dapat menganggap ini sebagai agen ReACT di sisi penyedia, di mana model menerima respons yang ditingkatkan berdasarkan alat yang dipanggilnya. Pendekatan ini mengurangi latensi dan biaya. Dalam kasus ini, Anda biasanya menyediakan pembungkus LangChain dengan alat kustom yang dibuat menggunakan SDK penyedia daripada yang dibangun dengan LangChain (yaitu, alat yang tidak mewarisi dari kelas `BaseTool`), yang berarti kode Anda tidak dapat dialihkan antar model.

# Menggabungkan alat ke dalam alur kerja

Sekarang setelah kita tahu cara membuat dan menggunakan alat, mari kita bahas bagaimana kita dapat menggabungkan paradigma pemanggilan alat lebih dalam ke dalam alur kerja yang kita kembangkan.

## Generasi terkendali

Di [Bab 3](Chapter_3.xhtml#_idTextAnchor049), kita mulai membahas generasi _terkendali_, ketika Anda ingin LLM mengikuti skema tertentu. Kita dapat meningkatkan alur kerja penguraian kita tidak hanya dengan membuat pengurai yang lebih canggih dan andal tetapi juga dengan lebih ketat dalam memaksa LLM untuk mematuhi skema tertentu. Memanggil alat memerlukan generasi terkendali karena payload yang dihasilkan harus mengikuti skema tertentu, tetapi kita dapat mengambil langkah mundur dan mengganti skema yang diharapkan dengan pemanggilan alat paksa yang mengikuti skema yang diharapkan. LangChain memiliki mekanisme bawaan untuk membantu dengan itu—LLM memiliki metode `with_structured_output` yang mengambil skema sebagai model Pydantic, mengubahnya menjadi alat, memanggil LLM dengan prompt yang diberikan dengan memaksanya untuk memanggil alat ini, dan mengurai keluaran dengan mengompilasi ke instance model Pydantic yang sesuai.

Nanti di bab ini, kita akan membahas agen rencana-dan-selesaikan, jadi mari kita mulai menyiapkan blok bangunan. Mari kita minta LLM kami untuk menghasilkan rencana untuk tindakan tertentu, tetapi alih-alih mengurai rencana, mari kita definisikan sebagai model Pydantic (`Plan` adalah daftar `Steps`):

```python
from pydantic import BaseModel, Field
class Step(BaseModel):
   """A step that is a part of the plan to solve the task."""
   step: str = Field(description="Description of the step")
class Plan(BaseModel):
   """A plan to solve the task."""
   steps: list[Step]
```

Ingatlah bahwa kami menggunakan model bersarang (satu bidang mereferensikan yang lain), tetapi LangChain akan mengkompilasi skema terunifikasi untuk kami. Mari kita satukan alur kerja sederhana dan jalankan:

```python
prompt = PromptTemplate.from_template(
   "Prepare a step-by-step plan to solve the given task.\n"
   "TASK:\n{task}\n"
)
result = (prompt | llm.with_structured_output(Plan)).invoke(
  "How to write a bestseller on Amazon about generative AI?")
```

Jika kita memeriksa keluaran, kita akan melihat bahwa kita mendapatkan model Pydantic sebagai hasilnya. Kita tidak perlu lagi mengurai keluaran; kita mendapatkan daftar langkah spesifik langsung (dan nanti, kita akan melihat bagaimana kita dapat menggunakannya lebih lanjut):

```python
assert isinstance(result, Plan)
print(f"Amount of steps: {len(result.steps)}")
for step in result.steps:
 print(step.step)
 break
```

```
>> Amount of steps: 21
**1. Idea Generation and Validation:**
```

### Generasi terkendali yang disediakan oleh vendor

Cara lain adalah tergantung vendor. Beberapa penyedia model dasar menawarkan parameter API tambahan yang dapat menginstruksikan model untuk menghasilkan keluaran terstruktur (biasanya, JSON atau enum). Anda dapat memaksa model untuk menggunakan generasi JSON dengan cara yang sama seperti di atas menggunakan `with_structured_output`, tetapi berikan argumen lain, `method="json_mode"` (dan periksa ulang bahwa penyedia model yang mendasarinya mendukung generasi terkendali sebagai JSON):

```python
plan_schema = {
   "type": "ARRAY",
   "items": {
       "type": "OBJECT",
         "properties": {
             "step": {"type": "STRING"},
         },
     },
}
query = "How to write a bestseller on Amazon about generative AI?"
result = (prompt | llm.with_structured_output(schema=plan_schema, method="json_mode")).invoke(query)
```

Perhatikan bahwa skema JSON tidak mengandung deskripsi bidang, oleh karena itu biasanya, prompt Anda harus lebih detail dan informatif. Tetapi sebagai keluaran, kita mendapatkan kamus Python yang berkualifikasi penuh:

```python
assert(isinstance(result, list))
print(f"Amount of steps: {len(result)}")
print(result[0])
```

```
>> Amount of steps: 10
{'step': 'Step 1: Define your niche and target audience. Generative AI is a broad topic. Focus on a specific area, like generative AI in marketing, art, music, or writing. Identify your ideal reader (such as  marketers, artists, developers).'}
```

Anda dapat menginstruksikan instance LLM langsung untuk mengikuti instruksi generasi terkendali. Perhatikan bahwa argumen dan fungsionalitas spesifik mungkin bervariasi dari satu penyedia model ke penyedia model lainnya (misalnya, model OpenAI menggunakan argumen `response_format`). Mari kita lihat cara menginstruksikan Gemini untuk mengembalikan JSON:

```python
from langchain_core.output_parsers import JsonOutputParser
llm_json = ChatVertexAI(
  model_name="gemini-2.0-flash", response_mime_type="application/json",
  response_schema=plan_schema)
result = (prompt | llm_json | JsonOutputParser()).invoke(query)
assert(isinstance(result, list))
```

Kami juga dapat meminta Gemini untuk mengembalikan enum—dengan kata lain, hanya satu nilai dari satu set nilai:

```python
from langchain_core.output_parsers import StrOutputParser
response_schema = {"type": "STRING", "enum": ["positive", "negative", "neutral"]}
prompt = PromptTemplate.from_template(
   "Classify the tone of the following customer's review:"
   "\n{review}\n"
)
review = "I like this movie!"
llm_enum = ChatVertexAI(model_name="gemini-1.5-pro-002", response_mime_type="text/x.enum", response_schema=response_schema)
result = (prompt | llm_enum | StrOutputParser()).invoke(review)
print(result)
```

```
>> positive
```

LangChain mengabstraksikan detail implementasi penyedia model dengan parameter `method="json_mode"` atau dengan mengizinkan `kwargs` kustom untuk diteruskan ke model. Beberapa kemampuan generasi terkendali adalah spesifik model. Periksa dokumentasi model Anda untuk tipe skema yang didukung, batasan, dan argumen.

## ToolNode

Untuk menyederhanakan pengembangan agen, LangGraph memiliki kemampuan bawaan seperti `ToolNode` dan `tool_conditions`. `ToolNode` memeriksa pesan terakhir di `messages` (Anda dapat mendefinisikan ulang nama kunci). Jika pesan ini berisi panggilan alat, itu memanggil alat yang sesuai dan memperbarui status. Di sisi lain, `tool_conditions` adalah tepi bersyarat yang memeriksa apakah `ToolNode` harus dipanggil (atau berakhir sebaliknya).

Sekarang kita dapat membangun mesin ReACT dalam beberapa menit:

```python
from langgraph.prebuilt import ToolNode, tools_condition
def invoke_llm(state: MessagesState):
   return {"messages": [llm_with_tools.invoke(state["messages"])]}
builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("tools", ToolNode([search, calculator]))
builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", tools_condition)
builder.add_edge("tools", "invoke_llm")
graph = builder.compile()
```

## Paradigma pemanggilan alat

Pemanggilan alat adalah paradigma desain yang sangat kuat yang membutuhkan perubahan dalam cara Anda mengembangkan aplikasi. Dalam banyak kasus, alih-alih melakukan putaran rekayasa prompt dan banyak upaya untuk meningkatkan prompt Anda, pikirkan apakah Anda bisa meminta model untuk memanggil alat.

Mari kita asumsikan kita sedang mengerjakan agen yang berurusan dengan pembatalan kontrak dan harus mengikuti logika bisnis tertentu. Pertama, kita perlu memahami tanggal mulai kontrak (dan berurusan dengan tanggal mungkin sulit!). Jika Anda mencoba membuat prompt yang dapat menangani kasus seperti ini dengan benar, Anda akan menyadari itu mungkin cukup sulit:

```python
examples = [
 "I signed my contract 2 years ago",
 "I started the deal with your company in February last year",
 "Our contract started on March 24th two years ago"
]
```

Sebagai gantinya, paksa model untuk memanggil alat (dan mungkin bahkan melalui agen ReACT!). Misalnya, kami memiliki dua alat yang sangat asli di Python—`date` dan `timedelta`:

```python
from datetime import date, timedelta
@tool
def get_date(year: int, month: int = 1, day: int = 1) -> date:
   """Returns a date object given year, month and day.
    Default month and day are 1 (January) and 1.
    Examples in YYYY-MM-DD format:
      2023-07-27 -> date(2023, 7, 27)
      2022-12-15 -> date(2022, 12, 15)
      March 2022 -> date(2022, 3)
      2021 -> date(2021)
  """
   return date(year, month, day).isoformat()
@tool
def time_difference(days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> date:
   """Returns a date given a difference in days, weeks, months and years relative to the current date.

    By default, days, weeks, months and years are 0.
    Examples:
      two weeks ago -> time_difference(weeks=2)
      last year -> time_difference(years=1)
    """
   dt = date.today() - timedelta(days=days, weeks=weeks)
   new_year = dt.year+(dt.month-months) // 12 - years
   new_month = (dt.month-months) % 12
   return dt.replace(year=new_year, month=new_month)
```

Sekarang ini bekerja dengan baik:

```python
from langchain_google_vertexai import ChatVertexAI
llm = ChatVertexAI(model="gemini-2.0-flash")
agent = create_react_agent(
   llm, [get_date, time_difference], prompt="Extract the starting date of a contract. Current year is 2025.")
for example in examples:
 result = agent.invoke({"messages": [("user", example)]})
 print(example, result["messages"][-1].content)
```

```
>> I signed my contract 2 years ago The contract started on 2023-02-07.
I started the deal with your company in February last year The contract started on 2024-02-01.
Our contract started on March 24th two years ago The contract started on 2023-03-24
```

Kami belajar bagaimana menggunakan alat, atau panggilan fungsi, untuk meningkatkan kinerja LLM pada tugas kompleks. Ini adalah salah satu pola arsitektur fundamental di balik agen—sekarang saatnya membahas apa itu agen.

# Apa itu agen?

Agen adalah salah satu topik terpanas AI generatif saat ini. Orang banyak membicarakan agen, tetapi ada banyak definisi berbeda tentang apa itu agen. LangChain sendiri mendefinisikan agen sebagai "_sistem yang menggunakan LLM untuk memutuskan alur kontrol aplikasi_." Sementara kami merasa ini adalah definisi yang bagus yang layak dikutip, itu melewatkan beberapa aspek.

Sebagai pengembang Python, Anda mungkin terbiasa dengan duck typing untuk menentukan perilaku objek dengan tes bebek yang disebut: "_Jika ia berjalan seperti bebek dan bersuara seperti bebek, maka itu pasti bebek_." Dengan konsep itu dalam pikiran, mari kita gambarkan beberapa properti agen dalam konteks AI generatif:

- Agen membantu pengguna menyelesaikan tugas non-deterministik kompleks tanpa diberikan algoritma eksplisit tentang bagaimana melakukannya. Agen lanjutan bahkan dapat bertindak atas nama pengguna.
- Untuk menyelesaikan tugas, agen biasanya melakukan beberapa langkah dan iterasi. Mereka _bernalar_ (menghasilkan informasi baru berdasarkan konteks yang tersedia), _bertindak_ (berinteraksi dengan lingkungan eksternal), _mengamati_ (menggabungkan umpan balik dari lingkungan eksternal), dan _berkomunikasi_ (berinteraksi dan/atau bekerja sama dengan agen lain atau manusia).
- Agen menggunakan LLM untuk penalaran (dan menyelesaikan tugas).
- Sementara agen memiliki otonomi tertentu (dan sampai batas tertentu, mereka bahkan mencari tahu apa cara terbaik untuk menyelesaikan tugas dengan berpikir dan belajar dari interaksi dengan lingkungan), saat menjalankan agen, kami masih ingin menjaga tingkat kendali tertentu atas alur eksekusi.

Mempertahankan kendali atas perilaku agen—alur kerja agen—adalah konsep inti di balik LangGraph. Sementara LangGraph memberi pengembang seperangkat blok bangunan yang kaya (seperti manajemen memori, pemanggilan alat, dan grafik siklik dengan kontrol kedalaman rekursi), pola desain utamanya berfokus pada mengelola alur dan tingkat otonomi yang LLM jalankan dalam mengeksekusi tugas. Mari kita mulai dengan contoh dan kembangkan agen kami.

## Agen rencana-dan-selesaikan

Apa yang biasanya kita lakukan sebagai manusia ketika kita memiliki tugas kompleks di depan kita? Kita merencanakan! Pada tahun 2023, Lei Want et al. menunjukkan bahwa prompt rencana-dan-selesaikan meningkatkan penalaran LLM. Ini juga telah ditunjukkan oleh banyak studi bahwa kinerja LLM cenderung memburuk seiring kompleksitas (khususnya, panjang dan jumlah instruksi) dari prompt meningkat.

Oleh karena itu, pola desain pertama yang harus diingat adalah _dekomposisi tugas_—untuk menguraikan tugas kompleks menjadi urutan yang lebih kecil, jaga prompt Anda sederhana dan fokus pada satu tugas, dan jangan ragu untuk menambahkan contoh ke prompt Anda. Dalam kasus kami, kami akan mengembangkan asisten penelitian.

Menghadapi tugas kompleks, mari pertama-tama minta LLM untuk membuat rencana detail untuk menyelesaikan tugas ini, dan kemudian gunakan LLM yang sama untuk mengeksekusi setiap langkah. Ingat, pada akhirnya, LLM menghasilkan token keluaran secara autoregresif berdasarkan token input. Pola sederhana seperti ReACT atau rencana-dan-selesaikan membantu kita menggunakan kemampuan penalaran implisit mereka dengan lebih baik.

Pertama, kita perlu mendefinisikan perencana kami. Tidak ada yang baru di sini; kami menggunakan blok bangunan yang sudah kami bahas—template prompt chat dan generasi terkendali dengan model Pydantic:

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
class Plan(BaseModel):
   """Plan to follow in future"""
   steps: list[str] = Field(
       description="different steps to follow, should be in sorted order"
   )
system_prompt_template = (
   "For the given task, come up with a step by step plan.\n"
   "This plan should involve individual tasks, that if executed correctly will "
   "yield the correct answer. Do not add any superfluous steps.\n"
   "The result of the final step should be the final answer. Make sure that each "
   "step has all the information needed - do not skip steps."
)
planner_prompt = ChatPromptTemplate.from_messages(
   [("system", system_prompt_template),
    ("user", "Prepare a plan how to solve the following task:\n{task}\n")])
planner = planner_prompt | ChatVertexAI(
   model_name="gemini-2.0-flash", temperature=1.0
).with_structured_output(Plan)
```

Untuk eksekusi langkah, mari gunakan agen ReACT dengan alat bawaan—pencarian DuckDuckGo, pengambil dari arXiv dan Wikipedia, dan alat `calculator` kustom kami yang kami kembangkan sebelumnya di bab ini:

```python
from langchain.agents import load_tools
tools = load_tools(
 tool_names=["ddg-search", "arxiv", "wikipedia"],
 llm=llm
) + [calculator_tool]
```

Selanjutnya, mari kita definisikan status alur kerja kami. Kita perlu melacak tugas awal dan rencana yang awalnya dihasilkan, dan mari tambahkan `past_steps` dan `final_response` ke status:

```python
class PlanState(TypedDict):
   task: str
   plan: Plan
   past_steps: Annotated[list[str], operator.add]
   final_response: str
   past_steps: list[str]
def get_current_step(state: PlanState) -> int:
 """Returns the number of current step to be executed."""
 return len(state.get("past_steps", []))

def get_full_plan(state: PlanState) -> str:
 """Returns formatted plan with step numbers and past results."""
 full_plan = []
 for i, step in enumerate(state["plan"]):
   full_step = f"# {i+1}. Planned step: {step}\n"
   if i < get_current_step(state):
     full_step += f"Result: {state['past_steps'][i]}\n"
   full_plan.append(full_step)
 return "\n".join(full_plan)
```

Sekarang, saatnya untuk mendefinisikan simpul dan tepi kami:

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
final_prompt = PromptTemplate.from_template(
   "You're a helpful assistant that has executed on a plan."
   "Given the results of the execution, prepare the final response.\n"
   "Don't assume anything\nTASK:\n{task}\n\nPLAN WITH RESUlTS:\n{plan}\n"
   "FINAL RESPONSE:\n"
)
async def _build_initial_plan(state: PlanState) -> PlanState:
 plan = await planner.ainvoke(state["task"])
 return {"plan": plan}
async def _run_step(state: PlanState) -> PlanState:
 plan = state["plan"]
 current_step = get_current_step(state)
 step = await execution_agent.ainvoke({"plan": get_full_plan(plan), "step": plan.steps[current_step], "task": state["task"]})
 return {"past_steps": [step["messages"][-1].content]}
async def _get_final_response(state: PlanState) -> PlanState:
 final_response = await (final_prompt | llm).ainvoke({"task": state["task"], "plan": get_full_plan(state)})
 return {"final_response": final_response}
def _should_continue(state: PlanState) -> Literal["run", "response"]:
 if get_current_step(plan) < len(state["plan"].steps):
   return "run"
 return "final_response"
```

Dan satukan grafik akhir:

```python
builder = StateGraph(PlanState)
builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_step)
builder.add_node("response", _get_final_response)
builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_conditional_edges("run", _should_continue)
builder.add_edge("response", END)
graph = builder.compile()
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

![Gambar 5.3: Alur kerja agen rencana-dan-selesaikan](Images/B32363_05_03.png)

Sekarang kita dapat menjalankan alur kerja:

```python
task = "Write a strategic one-pager of building an AI startup"
result = await graph.ainvoke({"task": task})
```

Anda dapat melihat keluaran lengkap di GitHub kami, dan kami mendorong Anda untuk bermain dengannya sendiri. Ini mungkin sangat menarik untuk menyelidiki apakah Anda menyukai hasilnya lebih dibandingkan dengan prompt LLM tunggal dengan tugas yang diberikan.

# Ringkasan

Dalam bab ini, kami mengeksplorasi cara meningkatkan LLM dengan mengintegrasikan alat dan pola desain untuk pemanggilan alat, termasuk pola ReACT. Kami mulai dengan membangun agen ReACT dari nol dan kemudian mendemonstrasikan cara membuat yang disesuaikan dengan hanya satu baris kode menggunakan LangGraph.

Selanjutnya, kami menyelami teknik lanjutan untuk generasi terkendali—menunjukkan cara memaksa LLM untuk memanggil alat apa pun atau spesifik, dan menginstruksikannya untuk mengembalikan respons dalam format terstruktur (seperti JSON, enum, atau model Pydantic). Dalam konteks itu, kami membahas metode `with_structured_output` LangChain, yang mengubah struktur data Anda menjadi skema alat, meminta model untuk memanggil alat, mengurai keluaran, dan mengkompilasinya ke instance Pydantic yang sesuai.

Akhirnya, kami membangun agen rencana-dan-selesaikan pertama kami dengan LangGraph, menerapkan semua konsep yang telah kami pelajari sejauh ini: pemanggilan alat, ReACT, keluaran terstruktur, dan banyak lagi. Di bab berikutnya, kami akan terus membahas cara mengembangkan agen dan melihat pola arsitektur yang lebih canggih.

# Pertanyaan

1. Apa manfaat utama menggunakan alat dengan LLM, dan mengapa mereka penting?
2. Bagaimana kelas ToolMessage LangChain memfasilitasi komunikasi antara LLM dan lingkungan eksternal?
3. Jelaskan pola ReACT. Apa dua langkah utamanya? Bagaimana itu meningkatkan kinerja LLM?
4. Bagaimana Anda akan mendefinisikan agen AI generatif? Bagaimana ini berhubungan atau berbeda dari definisi LangChain?
5. Jelaskan beberapa kelebihan dan kekurangan menggunakan metode with_structured_output dibandingkan menggunakan generasi terkendali langsung.
6. Bagaimana Anda dapat mendefinisikan alat kustom secara terprogram di LangChain?
7. Jelaskan tujuan metode Runnable.bind() dan bind_tools() di LangChain.
8. Bagaimana LangChain menangani kesalahan yang terjadi selama eksekusi alat? Opsi apa yang tersedia untuk mengonfigurasi perilaku ini?

# Berlangganan buletin mingguan kami

Berlangganan AI_Distilled, buletin untuk profesional, peneliti, dan inovator AI, di [https://packt.link/Q5UyU](Chapter_5.xhtml).

![Kode QR buletin](Images/Newsletter_QRcode1.jpg)
