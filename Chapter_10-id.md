# Masa Depan Model Generatif: Melampaui Penskalaan

Selama dekade terakhir, paradigma dominan dalam kemajuan AI telah berupa _penskalaan_—meningkatkan ukuran model (jumlah parameter), memperluas kumpulan data pelatihan, dan menerapkan lebih banyak sumber daya komputasi. Pendekatan ini telah memberikan peningkatan yang mengesankan, dengan setiap lompatan dalam ukuran model membawa kemampuan yang lebih baik. Namun, penskalaan saja menghadapi hasil yang semakin berkurang (diminishing returns) dan tantangan yang berkembang dalam hal keberlanjutan, aksesibilitas, dan mengatasi keterbatasan AI mendasar. Masa depan AI generatif terletak di luar penskalaan sederhana, dalam arsitektur yang lebih efisien, pendekatan khusus, dan sistem hibrida yang mengatasi keterbatasan saat ini sambil mendemokratisasi akses ke teknologi kuat ini.

Sepanjang buku ini, kita telah mengeksplorasi membangun aplikasi menggunakan model AI generatif. Fokus kami pada agen telah menjadi pusat, karena kami telah mengembangkan alat otonom yang dapat bernalar, merencanakan, dan mengeksekusi tugas di berbagai domain. Untuk pengembang dan ilmuwan data, kami telah mendemonstrasikan teknik termasuk integrasi alat, kerangka kerja penalaran berbasis agen, RAG, dan rekayasa petunjuk (prompt engineering) yang efektif—semua diimplementasikan melalui LangChain dan LangGraph. Saat kami menyimpulkan eksplorasi kami, tepat untuk mempertimbangkan implikasi teknologi ini dan ke mana bidang AI agenik yang berkembang cepat mungkin membawa kita selanjutnya. Oleh karena itu, dalam bab ini, kami akan merenungkan keterbatasan saat ini dari model generatif—bukan hanya teknis, tetapi tantangan sosial dan etika yang lebih besar yang mereka timbulkan. Kami akan melihat strategi untuk mengatasi masalah ini, dan mengeksplorasi di mana peluang nyata untuk penciptaan nilai berada—terutama ketika menyangkut menyesuaikan model untuk industri dan kasus penggunaan spesifik.

Kami juga akan mempertimbangkan apa arti AI generatif untuk pekerjaan, dan bagaimana hal itu dapat membentuk kembali seluruh sektor—dari bidang kreatif dan pendidikan hingga hukum, kedokteran, manufaktur, dan bahkan pertahanan. Akhirnya, kami akan menangani beberapa pertanyaan sulit seputar misinformasi, keamanan, privasi, dan keadilan—dan berpikir bersama tentang bagaimana teknologi ini harus diimplementasikan dan diatur di dunia nyata.

Area utama yang akan kita diskusikan dalam bab ini adalah:

- Keadaan terkini AI generatif
- Keterbatasan penskalaan dan alternatif yang muncul
- Transformasi ekonomi dan industri
- Implikasi sosial

## Keadaan terkini AI generatif

Seperti dibahas dalam buku ini, dalam beberapa tahun terakhir, model AI generatif telah mencapai pencapaian baru dalam menghasilkan konten seperti manusia di berbagai modalitas termasuk teks, gambar, audio, dan video. Model terkemuka seperti GPT-4o OpenAI, Claude 3.7 Sonnet Anthropic, Llama 3 Meta, dan Gemini 1.5 Pro serta 2.0 Google menunjukkan kefasihan mengesankan dalam generasi konten, baik tekstual atau seni visual kreatif.

Momen penting dalam pengembangan AI terjadi pada akhir 2024 dengan rilis model o1 OpenAI, diikuti segera oleh o3. Model-model ini mewakili pergeseran mendasar dalam kemampuan AI, terutama dalam domain yang memerlukan penalaran canggih. Tidak seperti peningkatan bertahap yang terlihat pada generasi sebelumnya, model-model ini menunjukkan lompatan luar biasa dalam kinerja. Mereka mencapai hasil tingkat medali emas dalam kompetisi Olimpiade Matematika Internasional dan mencocokkan kinerja tingkat PhD di seluruh masalah fisika, kimia, dan biologi.

Apa yang membedakan model baru seperti o1 dan o3 adalah pendekatan pemrosesan berulang (iterative) mereka yang dibangun di atas arsitektur transformer generasi sebelumnya. Model-model ini mengimplementasikan apa yang oleh peneliti digambarkan sebagai pola komputasi _rekursif_ yang memungkinkan beberapa lintasan pemrosesan atas informasi daripada mengandalkan hanya pada lintasan maju tunggal. Pendekatan ini memungkinkan model mengalokasikan sumber daya komputasi tambahan untuk masalah yang lebih menantang, meskipun ini tetap terikat oleh arsitektur mendasar dan paradigma pelatihan mereka. Sementara model-model ini menggabungkan beberapa mekanisme perhatian (attention) khusus untuk berbagai jenis masukan, mereka masih beroperasi dalam kendala jaringan saraf besar dan homogen daripada sistem yang benar-benar modular. Metodologi pelatihan mereka telah berkembang melampaui prediksi token berikutnya (next-token) sederhana untuk memasukkan optimalisasi untuk langkah-langkah penalaran menengah, meskipun pendekatan inti tetap didasarkan pada pengenalan pola statistik.

Munculnya model yang dipasarkan sebagai memiliki _kemampuan penalaran_ menunjukkan evolusi potensial dalam bagaimana sistem ini memproses informasi, meskipun keterbatasan signifikan tetap ada. Model-model ini menunjukkan kinerja yang lebih baik pada tugas penalaran terstruktur tertentu dan dapat mengikuti rantai pemikiran yang lebih eksplisit, terutama dalam domain yang terwakili dengan baik dalam data pelatihan mereka. Namun, seperti yang ditunjukkan oleh perbandingan dengan kognisi manusia, sistem ini terus berjuang dengan domain baru, pemahaman kausal, dan pengembangan konsep yang benar-benar baru. Ini mewakili kemajuan bertahap dalam bagaimana bisnis dapat memanfaatkan teknologi AI daripada pergeseran mendasar dalam kemampuan. Organisasi yang mengeksplorasi teknologi ini harus mengimplementasikan kerangka kerja pengujian yang ketat untuk mengevaluasi kinerja pada kasus penggunaan spesifik mereka, dengan perhatian khusus pada kasus tepi (edge cases) dan skenario yang memerlukan penalaran kausal sejati atau adaptasi domain.

Model dengan pendekatan penalaran yang ditingkatkan menunjukkan janji tetapi datang dengan keterbatasan penting yang harus menginformasikan implementasi bisnis:

- **Pendekatan analisis terstruktur**: Penelitian terbaru menunjukkan model ini dapat mengikuti pola penalaran multi-langkah untuk jenis masalah tertentu, meskipun penerapannya pada tantangan bisnis strategis tetap menjadi area eksplorasi aktif daripada kemampuan yang mapan.
- **Pertimbangan keandalan**: Sementara pendekatan penalaran langkah-demi-langkah menunjukkan janji pada beberapa tugas tolok ukur, penelitian menunjukkan teknik ini sebenarnya dapat memperparah kesalahan dalam konteks tertentu.
- **Sistem agen semi-otonom**: Model yang menggabungkan teknik penalaran dapat mengeksekusi beberapa tugas dengan intervensi manusia yang berkurang, tetapi implementasi saat ini memerlukan pemantauan dan pengaman (guardrails) yang cermat untuk mencegah propagasi kesalahan dan memastikan keselarasan dengan tujuan bisnis.

Yang sangat menonjol adalah peningkatan kemahiran dalam generasi kode, di mana model penalaran ini tidak hanya dapat menulis kode tetapi juga memahami, men-debug, dan meningkatkan secara berulang. Kemampuan ini mengarah ke masa depan di mana sistem AI berpotensi membuat dan mengeksekusi kode secara otonom, pada dasarnya memprogram diri mereka sendiri untuk memecahkan masalah baru atau beradaptasi dengan kondisi yang berubah—langkah mendasar menuju kecerdasan buatan yang lebih umum.

Aplikasi bisnis potensial dari model dengan pendekatan penalaran adalah signifikan, meskipun saat ini lebih aspirasional daripada diimplementasikan secara luas. Pengadopsi awal mengeksplorasi sistem di mana asisten AI dapat membantu menganalisis data pasar, mengidentifikasi masalah operasional potensial, dan meningkatkan dukungan pelanggan melalui pendekatan penalaran terstruktur. Namun, implementasi ini sebagian besar tetap eksperimental daripada sistem yang sepenuhnya otonom.

Sebagian besar penyebaran bisnis saat ini fokus pada tugas yang lebih sempit, terdefinisi dengan baik dengan pengawasan manusia daripada skenario sepenuhnya otonom yang kadang digambarkan dalam materi pemasaran. Sementara lab penelitian dan perusahaan teknologi terkemuka mendemonstrasikan prototipe yang menjanjikan, penyebaran luas sistem berbasis penalaran sejati untuk pengambilan keputusan bisnis kompleks tetap menjadi batas yang muncul daripada praktik yang mapan. Organisasi yang mengeksplorasi teknologi ini harus fokus pada program pilot terkontrol dengan metrik evaluasi cermat untuk menilai dampak bisnis nyata.

Untuk perusahaan yang mengevaluasi kemampuan AI, model penalaran mewakili langkah signifikan ke depan dalam membuat AI menjadi alat yang andal dan mampu untuk aplikasi bisnis bernilai tinggi. Kemajuan ini mengubah AI generatif dari terutama teknologi penciptaan konten menjadi sistem dukungan keputusan strategis yang mampu meningkatkan operasi bisnis inti.

Aplikasi praktis dari kemampuan penalaran ini membantu menjelaskan mengapa pengembangan model seperti o1 mewakili momen penting dalam evolusi AI. Seperti yang akan kita eksplorasi di bagian selanjutnya, implikasi dari kemampuan penalaran ini sangat bervariasi di berbagai industri, dengan beberapa sektor diposisikan untuk mendapat manfaat lebih segera daripada yang lain.

Apa yang membedakan model penalaran ini bukan hanya kinerjanya tetapi bagaimana mereka mencapainya. Sementara model sebelumnya berjuang dengan penalaran multi-langkah, sistem ini menunjukkan kemampuan untuk membangun rantai logis yang koheren, mengeksplorasi beberapa jalur solusi, mengevaluasi hasil menengah, dan membangun bukti kompleks. Evaluasi ekstensif mengungkapkan pola penalaran yang fundamentally berbeda dari model sebelumnya—menyerupai pendekatan pemecahan masalah yang disengaja dari penalar manusia ahli daripada pencocokan pola statistik.

Aspek paling signifikan dari model ini untuk diskusi kita tentang penskalaan adalah bahwa kemampuan mereka tidak dicapai terutama melalui peningkatan ukuran. Sebaliknya, mereka mewakili terobosan dalam arsitektur dan pendekatan pelatihan:

- **Arsitektur penalaran lanjutan** yang mendukung proses berpikir rekursif
- **Pembelajaran yang diawasi proses (process-supervised learning)** yang mengevaluasi dan memberi penghargaan pada langkah-langkah penalaran menengah, bukan hanya jawaban akhir
- **Alokasi komputasi waktu uji (test-time computation allocation)** yang memungkinkan model berpikir lebih lama tentang masalah sulit
- **Pembelajaran penguatan bermain sendiri (self-play reinforcement learning)** di mana model meningkat dengan bersaing melawan diri mereka sendiri

Perkembangan ini menantang hipotesis penskalaan sederhana dengan menunjukkan bahwa inovasi arsitektur kualitatif dan pendekatan pelatihan baru dapat menghasilkan peningkatan tidak kontinu dalam kemampuan. Mereka menunjukkan bahwa masa depan kemajuan AI mungkin lebih bergantung pada bagaimana model distrukturkan untuk berpikir daripada pada jumlah parameter mentah—tema yang akan kita eksplorasi lebih lanjut di bagian Keterbatasan penskalaan.

Berikut melacak kemajuan sistem AI di berbagai kemampuan relatif terhadap kinerja manusia selama periode 25 tahun. Kinerja manusia berfungsi sebagai garis dasar (ditentukan nol pada sumbu vertikal), sementara kinerja awal setiap kemampuan AI dinormalisasi ke -100. Bagan mengungkapkan lintasan dan garis waktu yang bervariasi untuk kemampuan AI berbeda yang mencapai dan melampaui kinerja tingkat manusia. Perhatikan kurva peningkatan yang sangat curam untuk penalaran prediktif, menunjukkan kemampuan ini tetap dalam fase kemajuan cepat daripada mendatar. Pemahaman bacaan, pemahaman bahasa, dan pengenalan gambar semuanya melintasi ambang kinerja manusia antara sekitar 2015 dan 2020, sementara pengenalan tulisan tangan dan ucapan mencapai pencapaian ini lebih awal.

Perbandingan antara kognisi manusia dan AI generatif mengungkapkan beberapa perbedaan mendasar yang bertahan meskipun kemajuan luar biasa antara 2022 dan 2025. Berikut adalah tabel yang merangkum kekuatan dan kekurangan utama AI generatif saat ini dibandingkan dengan kognisi manusia:

| Kategori                               | Kognisi Manusia                                                                                                                                                                                               | AI Generatif                                                                                                                                                                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pemahaman konseptual**               | Membentuk model kausal yang didasarkan pada pengalaman fisik dan sosial; membangun hubungan konsep bermakna di luar pola statistik                                                                            | Bergantung terutama pada pengenalan pola statistik tanpa pemahaman kausal sejati; dapat memanipulasi simbol dengan fasih tanpa pemahaman semantik yang lebih dalam                                                                     |
| **Pemrosesan faktual**                 | Mengintegrasikan pengetahuan dengan bias kognitif signifikan; rentan terhadap berbagai kesalahan penalaran sambil mempertahankan keandalan fungsional untuk bertahan hidup                                    | Menghasilkan informasi yang percaya diri tetapi sering dihalusinasi; berjuang membedakan informasi andal dari yang tidak andal meskipun ada augmentasi pengambilan (retrieval augmentation)                                            |
| **Pembelajaran dan penalaran adaptif** | Akuisisi lambat keterampilan kompleks tetapi sangat efisien sampel; mentransfer strategi di seluruh domain menggunakan pemikiran analogis; dapat menggeneralisasi dari beberapa contoh dalam konteks familiar | Membutuhkan kumpulan data masif untuk pelatihan awal; kemampuan penalaran sangat terikat oleh distribusi pelatihan; semakin mampu pembelajaran dalam konteks (in-context learning) tetapi berjuang dengan domain yang benar-benar baru |
| **Memori dan pelacakan status**        | Memori kerja terbatas (4-7 potongan); sangat baik dalam melacak status relevan meskipun kendala kapasitas; mengkompensasi dengan perhatian selektif                                                           | Secara teoretis jendela konteks tidak terbatas, tetapi kesulitan mendasar dengan pelacakan koheren dari status objek dan agen di seluruh skenario yang diperpanjang                                                                    |
| **Pemahaman sosial**                   | Secara alami mengembangkan model status mental orang lain melalui pengalaman yang diwujudkan (embodied); pemahaman intuitif tentang dinamika sosial dengan bakat individual yang bervariasi                   | Kapasitas terbatas untuk melacak status kepercayaan dan dinamika sosial yang berbeda; memerlukan penyempurnaan khusus untuk kemampuan teori pikiran dasar                                                                              |
| **Generasi kreatif**                   | Menghasilkan kombinasi baru yang melampaui pengalaman sebelumnya; inovasi didasarkan pada rekombinasi, tetapi dapat mendorong batas konseptual                                                                | Terbatas oleh distribusi pelatihan; menghasilkan variasi pada pola yang dikenal daripada konsep yang benar-benar baru                                                                                                                  |
| **Sifat arsitektural**                 | Organisasi modular, hierarkis dengan subsistem khusus; pemrosesan terdistribusi paralel dengan efisiensi energi luar biasa (~20 watt)                                                                         | Arsitektur yang sebagian besar homogen dengan spesialisasi fungsional terbatas; membutuhkan sumber daya komputasi masif untuk pelatihan dan inferensi                                                                                  |

Tabel 10.1: Perbandingan antara kognisi manusia dan AI generatif

Sementara sistem AI saat ini telah membuat kemajuan luar biasa dalam menghasilkan konten berkualitas tinggi di berbagai modalitas (gambar, video, teks koheren), mereka terus menunjukkan keterbatasan signifikan dalam kemampuan kognitif yang lebih dalam.

Penelitian terbaru menyoroti keterbatasan yang sangat mendalam dalam kecerdasan sosial. Studi Desember 2024 oleh Sclar et al. menemukan bahwa bahkan model terdepan seperti Llama-3.1 70B dan GPT-4o menunjukkan kinerja yang sangat buruk (serendah 0-9% akurasi) pada skenario **Teori Pikiran (Theory of Mind)** (**ToM**) yang menantang. Ketidakmampuan ini untuk memodelkan status mental orang lain, terutama ketika mereka berbeda dari informasi yang tersedia, mewakili celah mendasar antara kognisi manusia dan AI.

Menariknya, studi yang sama menemukan bahwa penyempurnaan bertarget (targeted fine-tuning) dengan skenario ToM yang dirancang dengan cermat menghasilkan peningkatan signifikan (+27 poin persentase), menunjukkan bahwa beberapa keterbatasan mungkin mencerminkan contoh pelatihan yang tidak memadai daripada kendala arsitektural yang tak teratasi. Pola ini meluas ke kemampuan lain—sementara penskalaan saja tidak cukup untuk mengatasi keterbatasan kognitif, pendekatan pelatihan khusus menunjukkan janji.

Celah dalam kemampuan pelacakan status sangat relevan. Meskipun secara teoretis jendela konteks tidak terbatas, sistem AI berjuang dengan melacak status objek dan pengetahuan agen secara koheren melalui skenario kompleks. Manusia, meskipun kapasitas memori kerja terbatas (biasanya 3-4 potongan menurut penelitian kognitif lebih baru), sangat baik dalam melacak status relevan melalui perhatian selektif dan strategi organisasi informasi yang efektif.

Sementara sistem AI telah membuat langkah mengesankan dalam integrasi multimodal (teks, gambar, audio, video), mereka masih kekurangan pemahaman lintas-modal yang mulus yang dikembangkan manusia secara alami. Demikian pula, dalam generasi kreatif, AI tetap terbatas oleh distribusi pelatihannya, menghasilkan variasi pada pola yang dikenal daripada konsep yang benar-benar baru.

Dari perspektif arsitektural, organisasi modular dan hierarkis otak manusia dengan subsistem khusus memungkinkan efisiensi energi luar biasa (~20 watt) dibandingkan dengan arsitektur AI yang sebagian besar homogen yang membutuhkan sumber daya komputasi masif. Selain itu, sistem AI dapat melanggengkan dan memperkuat bias yang ada dalam data pelatihan mereka, menimbulkan kekhawatiran etika di luar keterbatasan kinerja.

Perbedaan ini menunjukkan bahwa sementara kemampuan tertentu mungkin meningkat melalui data dan teknik pelatihan yang lebih baik, yang lain mungkin memerlukan inovasi arsitektural yang lebih mendasar untuk menjembatani kesenjangan antara pencocokan pola statistik dan pemahaman sejati.

Terlepas dari kemajuan mengesankan dalam AI generatif, celah mendasar tetap ada antara kognisi manusia dan AI di berbagai dimensi. Yang paling kritis, AI kekurangan:

- Landasan dunia nyata untuk pengetahuan
- Fleksibilitas adaptif di berbagai konteks
- Pemahaman yang benar-benar terintegrasi di bawah kefasihan permukaan
- Pemrosesan yang efisien energi
- Kesadaran sosial dan kontekstual

Keterbatasan ini bukan masalah terisolasi tetapi aspek yang saling berhubungan dari tantangan mendasar yang sama dalam mengembangkan kecerdasan buatan yang benar-benar seperti manusia. Seiring dengan kemajuan teknis, lanskap regulasi untuk AI berkembang dengan cepat, menciptakan pasar global yang kompleks. Undang-Undang AI Uni Eropa, diimplementasikan pada 2024, telah menciptakan persyaratan ketat yang menunda atau membatasi ketersediaan beberapa alat AI di pasar Eropa. Misalnya, Meta AI tersedia di Prancis hanya pada 2025, dua tahun setelah rilis AS, karena tantangan kepatuhan regulasi. Perbedaan regulasi yang berkembang ini menambah dimensi lain pada evolusi AI di luar penskalaan teknis, karena perusahaan harus menyesuaikan penawaran mereka untuk memenuhi persyaratan hukum yang bervariasi sambil mempertahankan kemampuan kompetitif.

## Keterbatasan penskalaan dan alternatif yang muncul

Memahami keterbatasan paradigma penskalaan dan alternatif yang muncul sangat penting bagi siapa pun yang membangun atau mengimplementasikan sistem AI saat ini. Sebagai pengembang dan pemangku kepentingan, mengenali di mana hasil yang semakin berkurang (diminishing returns) mulai terjadi membantu menginformasikan keputusan investasi yang lebih baik, pilihan teknologi, dan strategi implementasi. Pergeseran di luar penskalaan mewakili tantangan dan peluang—tantangan untuk memikirkan kembali bagaimana kita memajukan kemampuan AI, dan peluang untuk menciptakan sistem yang lebih efisien, dapat diakses, dan khusus. Dengan mengeksplorasi keterbatasan dan alternatif ini, pembaca akan lebih siap untuk menavigasi lanskap AI yang berkembang, membuat keputusan arsitektur yang tepat, dan mengidentifikasi jalur paling menjanjikan ke depan untuk kasus penggunaan spesifik mereka.

### Hipotesis penskalaan ditantang

Waktu penggandaan saat ini dalam komputasi pelatihan model yang sangat besar adalah sekitar 8 bulan, melampaui hukum penskalaan mapan seperti Hukum Moore (kepadatan transistor pada biaya meningkat dengan laju saat ini sekitar 18 bulan) dan Hukum Rock (biaya perangkat keras seperti GPU dan TPU setengah setiap 4 tahun).

Menurut dokumen _Situational Awareness_ Leopold Aschenbrenner dari Juni 2024, komputasi pelatihan AI telah meningkat sekitar 4,6x per tahun sejak 2010, sementara FLOP/s GPU hanya meningkat sekitar 1,35x per tahun. Peningkatan algoritmik memberikan peningkatan kinerja sekitar 3x per tahun. Kecepatan luar biasa dari penskalaan komputasi ini mencerminkan perlombaan senjata yang belum pernah terjadi sebelumnya dalam pengembangan AI, jauh melampaui norma penskalaan semikonduktor tradisional.

Gemini Ultra diperkirakan menggunakan sekitar 5 × 10^25 FLOP dalam pelatihan akhirnya, menjadikannya (pada saat penulisan ini) kemungkinan model paling intensif komputasi yang pernah dilatih. Secara bersamaan, kumpulan data pelatihan model bahasa telah tumbuh sekitar 3,0x per tahun sejak 2010, menciptakan persyaratan data masif.

Pada 2024-2025, pergeseran signifikan dalam perspektif telah terjadi mengenai _hipotesis penskalaan_—gagasan bahwa hanya dengan meningkatkan ukuran model, data, dan komputasi akan secara tak terelakkan mengarah pada **kecerdasan buatan umum (artificial general intelligence)** (**AGI**). Meskipun investasi masif (diperkirakan hampir setengah triliun dolar) dalam pendekatan ini, bukti menunjukkan bahwa penskalaan saja mencapai hasil yang semakin berkurang (diminishing returns) karena beberapa alasan:

- Pertama, kinerja telah mulai mendatar. Meskipun peningkatan besar dalam ukuran model dan komputasi pelatihan, tantangan mendasar seperti halusinasi, penalaran yang tidak andal, dan ketidakakuratan faktual bertahan bahkan dalam model terbesar. Rilis terkenal seperti Grok 3 (dengan 15x komputasi pendahulunya) masih menunjukkan kesalahan dasar dalam penalaran, matematika, dan informasi faktual.
- Kedua, lanskap kompetitif telah bergeser secara dramatis. Keunggulan teknologi yang pernah jelas dari perusahaan seperti OpenAI telah terkikis, dengan 7-10 model tingkat GPT-4 sekarang tersedia di pasar. Perusahaan China seperti DeepSeek telah mencapai kinerja sebanding dengan komputasi yang jauh lebih sedikit (sekecil 1/50 dari biaya pelatihan), menantang gagasan bahwa keunggulan sumber daya masif diterjemahkan menjadi keunggulan teknologi yang tak teratasi.
- Ketiga, ketidakberlanjutan ekonomi menjadi jelas. Pendekatan penskalaan telah menyebabkan biaya besar tanpa pendapatan proporsional. Perang harga telah pecah saat pesaing dengan kemampuan serupa saling menekan, memampatkan margin dan mengikis kasus ekonomi untuk model yang semakin besar.
- Akhirnya, pengakuan industri terhadap keterbatasan ini telah tumbuh. Tokoh industri kunci, termasuk CEO Microsoft Satya Nadella dan investor terkemuka seperti Marc Andreessen, telah secara publik mengakui bahwa hukum penskalaan mungkin mencapai batas, mirip dengan bagaimana Hukum Moore akhirnya melambat dalam pembuatan chip.

### Teknologi besar vs. perusahaan kecil

Kebangkitan AI open source telah sangat transformatif dalam lanskap yang berubah ini. Proyek seperti Llama, Mistral, dan lainnya telah mendemokratisasi akses ke model fondasi (foundation models) yang kuat, memungkinkan perusahaan kecil untuk membangun, menyempurnakan, dan menyebarkan LLM mereka sendiri tanpa investasi masif yang sebelumnya diperlukan. Ekosistem open source ini telah menciptakan tanah subur untuk inovasi di mana model khusus, domain-spesifik yang dikembangkan oleh tim lebih kecil dapat mengungguli model umum dari raksasa teknologi dalam aplikasi spesifik, lebih lanjut mengikis keunggulan skala saja.

Beberapa perusahaan kecil telah berhasil menunjukkan dinamika ini. Cohere, dengan tim yang jauh lebih kecil dari Google atau OpenAI, telah mengembangkan model berfokus perusahaan khusus yang cocok atau melampaui pesaing yang lebih besar dalam aplikasi bisnis melalui metodologi pelatihan inovatif yang berfokus pada pengikuti instruksi dan keandalan. Demikian pula, Anthropic mencapai kinerja perintah dengan model Claude yang sering mengungguli pesaing yang lebih besar dalam tolok ukur penalaran dan keamanan dengan menekankan pendekatan AI konstitusional daripada hanya skala. Di ranah open-source, Mistral AI berulang kali menunjukkan bahwa model lebih kecil mereka yang dirancang dengan cermat dapat mencapai kinerja kompetitif dengan model berkali-kali ukuran mereka.

Apa yang semakin jelas adalah bahwa parit teknologi yang pernah jelas dinikmati oleh perusahaan Teknologi Besar (Big Tech) dengan cepat terkikis. Lanskap kompetitif telah bergeser secara dramatis pada 2024-2025.

Beberapa model mampu telah muncul. Di mana OpenAI pernah berdiri sendiri dengan ChatGPT dan GPT-4, sekarang ada 7-10 model sebanding tersedia di pasar dari perusahaan seperti Anthropic, Google, Meta, Mistral, dan DeepSeek, secara signifikan mengurangi keunikan dan keunggulan teknologi yang dirasakan OpenAI.

Perang harga dan komoditisasi telah meningkat. Saat kemampuan telah disamakan, penyedia telah terlibat dalam pemotongan harga agresif. OpenAI berulang kali menurunkan harga sebagai respons terhadap tekanan kompetitif, terutama dari perusahaan China yang menawarkan kemampuan serupa dengan biaya lebih rendah.

Pemain non-tradisional telah menunjukkan pengejaran cepat. Perusahaan seperti DeepSeek dan ByteDance telah mencapai kualitas model sebanding dengan biaya pelatihan yang jauh lebih rendah, menunjukkan bahwa metodologi pelatihan inovatif dapat mengatasi kesenjangan sumber daya. Selain itu, siklus inovasi telah memendek secara signifikan. Kemajuan teknis baru dicocokkan atau dilampaui dalam beberapa minggu atau bulan daripada tahun, membuat keunggulan teknologi apa pun semakin sementara.

Melihat lanskap adopsi teknologi, kita dapat mempertimbangkan dua skenario utama untuk implementasi AI. Dalam skenario terpusat, AI generatif dan LLM terutama dikembangkan dan dikendalikan oleh perusahaan teknologi besar yang berinvestasi besar dalam perangkat keras komputasi yang diperlukan, penyimpanan data, dan talenta AI/ML khusus. Entitas ini menghasilkan model berpemilik umum yang sering dibuat dapat diakses oleh pelanggan melalui layanan cloud atau API, tetapi solusi satu-untuk-semua ini mungkin tidak selaras sempurna dengan persyaratan setiap pengguna atau organisasi.

Sebaliknya, dalam skenario layanan mandiri (self-service), perusahaan atau individu mengambil tugas menyempurnakan model AI mereka sendiri. Pendekatan ini memungkinkan mereka membuat model yang disesuaikan dengan kebutuhan spesifik dan data kepemilikan pengguna, memberikan fungsionalitas yang lebih ditargetkan dan relevan. Saat biaya menurun untuk komputasi, penyimpanan data, dan talenta AI, penyempurnaan kustom dari model khusus sudah layak untuk perusahaan kecil dan menengah.

Lanskap hibrida kemungkinan akan muncul di mana kedua pendekatan memenuhi peran berbeda berdasarkan kasus penggunaan, sumber daya, keahlian, dan pertimbangan privasi. Perusahaan besar mungkin terus unggul dalam menyediakan model khusus industri, sementara entitas kecil dapat semakin menyempurnakan model mereka sendiri untuk memenuhi permintaan ceruk.

Jika alat yang kuat muncul untuk menyederhanakan dan mengotomatisasi pengembangan AI, model generatif kustom bahkan mungkin layak untuk pemerintah lokal, kelompok komunitas, dan individu untuk mengatasi tantangan hiper-lokal. Sementara perusahaan teknologi besar saat ini mendominasi penelitian dan pengembangan AI generatif, entitas kecil pada akhirnya mungkin paling diuntungkan dari teknologi ini.

### Alternatif yang muncul untuk penskalaan murni

Saat keterbatasan penskalaan menjadi lebih jelas, beberapa pendekatan alternatif mendapatkan daya tarik. Banyak perspektif ini tentang melampaui penskalaan murni terinspirasi oleh makalah berpengaruh Juni 2024 Leopold Aschenbrenner _Situational Awareness: The Decade Ahead_ ([https://situational-awareness.ai/](https://situational-awareness.ai/)), yang memberikan analisis komprehensif tentang tren penskalaan AI dan keterbatasannya sambil mengeksplorasi paradigma alternatif untuk kemajuan. Pendekatan ini dapat diorganisir menjadi tiga paradigma utama. Mari kita lihat masing-masing.

#### Penskalaan naik (pendekatan tradisional)

Pendekatan tradisional untuk kemajuan AI telah berpusat pada penskalaan naik—mengejar kemampuan lebih besar melalui model yang lebih besar, lebih banyak komputasi, dan kumpulan data yang lebih besar. Paradigma ini dapat dipecah menjadi beberapa komponen kunci:

- **Meningkatkan ukuran dan kompleksitas model**: Pendekatan dominan sejak 2017 adalah menciptakan jaringan saraf yang semakin besar dengan lebih banyak parameter. GPT-3 berkembang menjadi 175 miliar parameter, sementara model yang lebih baru seperti GPT-4 dan Gemini Ultra diperkirakan memiliki beberapa triliun parameter efektif. Setiap peningkatan ukuran umumnya menghasilkan peningkatan kemampuan di berbagai tugas.
- **Memperluas sumber daya komputasi**: Melatih model masif ini memerlukan infrastruktur komputasi yang sangat besar. Pelatihan AI terbesar sekarang mengonsumsi sumber daya yang sebanding dengan pusat data kecil, dengan penggunaan listrik, persyaratan pendinginan, dan kebutuhan perangkat keras khusus yang membuatnya di luar jangkauan semua kecuali organisasi terbesar. Satu run pelatihan untuk model terdepan dapat menelan biaya lebih dari $100 juta.
- **Mengumpulkan kumpulan data besar**: Saat model tumbuh, begitu pula rasa lapar mereka akan data pelatihan. Model terkemuka dilatih pada triliun token, pada dasarnya mengonsumsi banyak teks berkualitas tinggi yang tersedia di internet, buku, dan kumpulan data khusus. Pendekatan ini memerlukan pipeline pemrosesan data canggih dan infrastruktur penyimpanan signifikan.
- **Keterbatasan menjadi jelas**: Sementara pendekatan ini mendominasi pengembangan AI hingga saat ini dan menghasilkan hasil luar biasa, ia menghadapi tantangan yang meningkat dalam hal hasil investasi yang semakin berkurang (diminishing returns), keberlanjutan ekonomi, dan hambatan teknis yang tidak dapat diatasi oleh penskalaan saja.

#### Penskalaan turun (inovasi efisiensi)

Paradigma efisiensi berfokus pada mencapai lebih dengan kurang melalui beberapa teknik kunci:

- **Kuantisasi (Quantization)** mengonversi model ke presisi lebih rendah dengan mengurangi ukuran bit dari bobot dan aktivasi. Teknik ini dapat mengompres kinerja model besar ke faktor bentuk yang lebih kecil, secara dramatis mengurangi persyaratan komputasi dan penyimpanan.
- **Distilasi model (Model distillation)** mentransfer pengetahuan dari model "guru" besar ke model "siswa" yang lebih kecil dan lebih efisien, memungkinkan penyebaran pada perangkat keras yang lebih terbatas.
- **Arsitektur yang diperkuat memori (Memory-augmented architectures)** mewakili pendekatan terobosan. Penelitian Meta FAIR Desember 2024 tentang lapisan memori (memory layers) menunjukkan bagaimana meningkatkan kemampuan model tanpa peningkatan proporsional dalam persyaratan komputasi. Dengan mengganti beberapa jaringan feed-forward dengan lapisan memori kunci-nilai (key-value) yang dapat dilatih yang diskalakan ke 128 miliar parameter, peneliti mencapai peningkatan lebih dari 100% dalam akurasi faktual sambil juga meningkatkan kinerja pada tugas pengkodean dan pengetahuan umum. Luar biasa, model yang diperkuat memori ini mencocokkan kinerja model padat (dense) yang dilatih dengan 4x lebih banyak komputasi, secara langsung menantang asumsi bahwa lebih banyak komputasi adalah satu-satunya jalan menuju kinerja yang lebih baik. Pendekatan ini secara khusus menargetkan keandalan faktual—mengatasi masalah halusinasi yang telah bertahan meskipun peningkatan skala dalam arsitektur tradisional.
- **Model khusus (Specialized models)** menawarkan alternatif lain untuk sistem tujuan umum. Daripada mengejar kecerdasan umum melalui skala, model yang difokuskan yang disesuaikan dengan domain spesifik sering memberikan kinerja yang lebih baik dengan biaya lebih rendah. Seri Phi Microsoft, sekarang maju ke phi-3 (April 2024), menunjukkan bagaimana kurasi data yang cermat dapat secara dramatis mengubah hukum penskalaan. Sementara model seperti GPT-4 dilatih pada kumpulan data heterogen yang luas, seri Phi mencapai kinerja luar biasa dengan model yang jauh lebih kecil dengan berfokus pada data berkualitas tinggi seperti buku teks.

#### Penskalaan keluar (pendekatan terdistribusi)

Paradigma terdistribusi ini mengeksplorasi cara memanfaatkan jaringan model dan sumber daya komputasi.

**Komputasi waktu uji (Test-time compute)** menggeser fokus dari melatih model yang lebih besar ke mengalokasikan lebih banyak komputasi selama waktu inferensi. Ini memungkinkan model untuk _bernalar_ melalui masalah lebih teliti. Pendekatan Mind Evolution Google DeepMind mencapai tingkat keberhasilan lebih dari 98% pada tugas perencanaan kompleks tanpa memerlukan model yang lebih besar, menunjukkan kekuatan strategi pencarian evolusioner selama inferensi. Pendekatan ini mengonsumsi tiga juta token karena petunjuk yang sangat panjang, dibandingkan dengan 9.000 token untuk operasi Gemini normal, tetapi mencapai hasil yang jauh lebih baik.

Kemajuan terbaru dalam kemampuan penalaran telah melampaui generasi token autoregresif sederhana dengan memperkenalkan konsep _pemikiran_—urutan token yang mewakili langkah-langkah menengah dalam proses penalaran. Pergeseran paradigma ini memungkinkan model meniru penalaran manusia kompleks melalui pendekatan pencarian pohon (tree search) dan pemikiran reflektif. Penelitian menunjukkan bahwa mendorong model untuk berpikir dengan lebih banyak token selama inferensi waktu uji secara signifikan meningkatkan akurasi penalaran.

Beberapa pendekatan telah muncul untuk memanfaatkan wawasan ini: Pengawasan berbasis proses (Process-based supervision), di mana model menghasilkan rantai penalaran langkah-demi-langkah dan menerima penghargaan pada langkah menengah. Teknik **Pencarian Pohon Monte Carlo (Monte Carlo Tree Search)** (**MCTS**) yang mengeksplorasi beberapa jalur penalaran untuk menemukan solusi optimal, dan model revisi (revision models) yang dilatih untuk memecahkan masalah secara berulang, menyempurnakan upaya sebelumnya.

Misalnya, makalah 2025 rStar-Math (_rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking_) menunjukkan bahwa model dapat mencapai kemampuan penalaran yang sebanding dengan o1 OpenAI tanpa distilasi dari model superior, melainkan memanfaatkan "pemikiran mendalam (deep thinking)" melalui MCTS yang dipandu oleh model penghargaan proses berbasis SLM. Ini mewakili pendekatan yang fundamentally berbeda untuk meningkatkan kemampuan AI daripada metode penskalaan tradisional.

**RAG** membumi (ground) keluaran model dalam sumber pengetahuan eksternal, yang membantu mengatasi masalah halusinasi lebih efektif daripada hanya meningkatkan ukuran model. Pendekatan ini memungkinkan bahkan model yang lebih kecil untuk mengakses informasi yang akurat dan terkini tanpa harus mengkodekannya semua dalam parameter.

**Mekanisme memori lanjutan (Advanced memory mechanisms)** telah menunjukkan hasil yang menjanjikan. Inovasi terbaru seperti lapisan memori Meta FAIR dan model memori saraf Titans Google menunjukkan kinerja superior sambil secara dramatis mengurangi persyaratan komputasi. Lapisan memori Meta menggunakan mekanisme pencarian kunci-nilai (key-value) yang dapat dilatih untuk menambahkan parameter tambahan ke model tanpa meningkatkan FLOP. Mereka meningkatkan akurasi faktual lebih dari 100% pada tolok ukur QA faktual sambil juga meningkatkan kinerja pada tugas pengkodean dan pengetahuan umum. Lapisan memori ini dapat diskalakan ke 128 miliar parameter dan telah dilatih sebelumnya hingga 1 triliun token.

Pendekatan inovatif lain dalam paradigma ini termasuk:

- **Model Memori Perhatian Saraf (Neural Attention Memory Models) (NAMMs)** meningkatkan kinerja dan efisiensi transformer tanpa mengubah arsitekturnya. NAMMs dapat memotong konteks masukan menjadi sebagian kecil dari ukuran asli sambil meningkatkan kinerja sebesar 11% pada LongBench dan memberikan peningkatan 10 kali lipat pada InfiniteBench. Mereka telah menunjukkan transferabilitas nol-shot (zero-shot transferability) ke arsitektur transformer baru dan modalitas masukan.
- **Pemodelan tingkat konsep (Concept-level modeling)**, seperti yang terlihat dalam Model Konsep Besar (Large Concept Models) Meta, beroperasi pada tingkat abstraksi yang lebih tinggi daripada token, memungkinkan pemrosesan yang lebih efisien. Alih-alih beroperasi pada token diskrit, LCM melakukan komputasi dalam ruang embedding berdimensi tinggi yang mewakili unit makna abstrak (konsep), yang sesuai dengan kalimat atau ucapan. Pendekatan ini secara inheren agnostik modalitas (modality-agnostic), mendukung lebih dari 200 bahasa dan banyak modalitas, termasuk teks dan ucapan.
- **Peningkatan berpusat penglihatan (Vision-centric enhancements)** seperti OLA-VLM mengoptimalkan model multimodal khusus untuk tugas visual tanpa memerlukan banyak pengkode visual (visual encoders). OLA-VLM meningkatkan kinerja di atas model dasar hingga 8,7% dalam tugas estimasi kedalaman dan mencapai skor 45,4% mIoU untuk tugas segmentasi (dibandingkan dengan dasar 39,3%).

Pergeseran ini menunjukkan bahwa masa depan pengembangan AI mungkin tidak didominasi hanya oleh organisasi dengan sumber daya komputasi terbanyak. Sebaliknya, inovasi dalam metodologi pelatihan, desain arsitektur, dan spesialisasi strategis dapat menentukan keunggulan kompetitif dalam fase berikutnya dari pengembangan AI.

### Evolusi kualitas data pelatihan

Evolusi kualitas data pelatihan telah menjadi semakin canggih dan mengikuti tiga perkembangan kunci. Pertama, model terkemuka menemukan bahwa buku memberikan keunggulan penting dibandingkan konten yang diambil dari web (web-scraped). GPT-4 ditemukan telah banyak menghafal karya sastra, termasuk seri _Harry Potter_, _Nineteen Eighty-Four_ Orwell, dan trilogi _The Lord of the Rings_—sumber dengan narasi koheren, struktur logis, dan bahasa halus yang sering kurang dalam konten web. Ini membantu menjelaskan mengapa model awal dengan akses ke korpus buku sering mengungguli model yang lebih besar yang dilatih terutama pada data web.

Kedua, kurasi data telah berkembang menjadi pendekatan bertingkat:

- **Kumpulan data emas (Golden datasets)**: Koleksi yang dibuat oleh ahli subjek tradisional yang mewakili standar kualitas tertinggi
- **Kumpulan data perak (Silver datasets)**: Konten yang dihasilkan LLM yang meniru instruksi tingkat ahli, memungkinkan penskalaan masif contoh pelatihan
- **Kumpulan data super emas (Super golden datasets)**: Koleksi yang divalidasi secara ketat yang dikurasi oleh berbagai ahli dengan beberapa lapisan verifikasi
- **Data penalaran sintetis (Synthetic reasoning data)**: Kumpulan data yang dihasilkan khusus yang berfokus pada pendekatan pemecahan masalah langkah-demi-langkah

Ketiga, penilaian kualitas telah menjadi semakin canggih. Pipeline persiapan data modern menggunakan beberapa tahap penyaringan, deteksi kontaminasi, deteksi bias, dan penilaian kualitas. Peningkatan ini secara dramatis telah mengubah hukum penskalaan tradisional—model 7 miliar parameter yang dilatih dengan baik dengan kualitas data luar biasa sekarang dapat mengungguli model 175 miliar parameter sebelumnya pada tugas penalaran kompleks.

Pendekatan yang berpusat pada data ini mewakili alternatif mendasar untuk penskalaan parameter murni, menunjukkan bahwa masa depan AI mungkin milik model yang lebih efisien dan khusus yang dilatih pada data yang tepat ditargetkan daripada sistem tujuan umum yang sangat besar yang dilatih pada semua yang tersedia.

Tantangan yang muncul untuk kualitas data adalah prevalensi yang semakin besar dari konten yang dihasilkan AI di seluruh internet. Saat sistem AI generatif menghasilkan lebih banyak teks, gambar, dan kode yang muncul online, model masa depan yang dilatih pada data ini akan semakin belajar dari keluaran AI lain daripada konten asli yang dibuat manusia. Ini menciptakan potensi loop umpan balik yang pada akhirnya dapat mengarah pada kinerja yang mendatar (plateauing), karena model mulai memperkuat pola, keterbatasan, dan bias yang ada dalam generasi AI sebelumnya daripada belajar dari contoh manusia yang segar. Fenomena _jenuh data AI (AI data saturation)_ ini menyoroti pentingnya terus mengkurasi konten berkualitas tinggi dan terverifikasi yang dibuat manusia untuk melatih model masa depan.

### Demokratisasi melalui kemajuan teknis

Biaya pelatihan model AI yang menurun dengan cepat mewakili pergeseran signifikan dalam lanskap, memungkinkan partisipasi yang lebih luas dalam penelitian dan pengembangan AI mutakhir. Beberapa faktor berkontribusi pada tren ini, termasuk optimalisasi rezim pelatihan, peningkatan kualitas data, dan pengenalan arsitektur model baru.

Berikut adalah teknik dan pendekatan kunci yang membuat AI generatif lebih mudah diakses dan efektif:

- **Arsitektur model yang disederhanakan (Simplified model architectures)**: Desain model yang dirampingkan untuk manajemen yang lebih mudah, interpretabilitas yang lebih baik, dan biaya komputasi yang lebih rendah
- **Generasi data sintetis (Synthetic data generation)**: Data pelatihan buatan yang menambah (augment) kumpulan data sambil menjaga privasi
- **Distilasi model (Model distillation)**: Transfer pengetahuan dari model besar ke model yang lebih kecil dan lebih efisien untuk penyebaran yang mudah
- **Mesin inferensi yang dioptimalkan (Optimized inference engines)**: Kerangka kerja perangkat lunak yang meningkatkan kecepatan dan efisiensi mengeksekusi model AI pada perangkat keras tertentu
- **Akselerator perangkat keras AI khusus (Dedicated AI hardware accelerators)**: Perangkat keras khusus seperti GPU dan TPU yang secara dramatis mempercepat komputasi AI
- **Data open-source dan sintetis (Open-source and synthetic data)**: Kumpulan data publik berkualitas tinggi yang memungkinkan kolaborasi dan meningkatkan privasi sambil mengurangi bias
- **Pembelajaran terfederasi (Federated learning)**: Pelatihan pada data terdesentralisasi untuk meningkatkan privasi sambil mendapat manfaat dari sumber yang beragam
- **Multimodalitas (Multimodality)**: Integrasi bahasa dengan gambar, video, dan modalitas lain dalam model teratas

Di antara kemajuan teknis yang membantu menurunkan biaya, teknik kuantisasi telah muncul sebagai kontributor penting. Kumpulan data dan teknik open-source seperti generasi data sintetis lebih lanjut mendemokratisasi akses ke pelatihan AI dengan memberikan pengembangan model yang berkualitas tinggi dan efisien data dan menghilangkan beberapa ketergantungan pada kumpulan data kepemilikan yang sangat besar. Inisiatif open-source berkontribusi pada tren dengan menyediakan platform yang hemat biaya dan kolaboratif untuk inovasi.

Inovasi ini secara kolektif menurunkan hambatan yang sejauh ini menghambat adopsi AI generatif dunia nyata dalam beberapa cara penting:

- Hambatan keuangan dikurangi dengan mengompres kinerja model besar ke faktor bentuk yang jauh lebih kecil melalui kuantisasi dan distilasi
- Pertimbangan privasi dapat berpotensi ditangani melalui teknik data sintetis, meskipun implementasi pembelajaran terfederasi yang andal dan dapat direproduksi untuk LLM khususnya tetap menjadi area penelitian yang sedang berlangsung daripada metodologi yang terbukti
- Keterbatasan akurasi yang menghambat model kecil diringankan dengan membumi (grounding) generasi dengan informasi eksternal
- Perangkat keras khusus secara signifikan mempercepat throughput sementara perangkat lunak yang dioptimalkan memaksimalkan efisiensi infrastruktur yang ada

Dengan mendemokratisasi akses dengan menangani kendala seperti biaya, keamanan, dan keandalan, pendekatan ini membuka manfaat untuk audiens yang sangat diperluas, mengarahkan kreativitas generatif dari konsentrasi sempit menuju memberdayakan bakat manusia yang beragam.

Lanskap bergeser dari fokus pada ukuran model mentah dan komputasi kasar (brute-force) ke pendekatan yang cerdas dan bernuansa yang memaksimalkan efisiensi komputasi dan kemanjuran model. Dengan kuantisasi dan teknik terkait menurunkan hambatan, kita siap untuk era pengembangan AI yang lebih beragam dan dinamis di mana kekayaan sumber daya bukan satu-satunya penentu kepemimpinan dalam inovasi AI.

### Hukum penskalaan baru untuk fase pasca-pelatihan

Tidak seperti penskalaan pra-pelatihan tradisional, di mana peningkatan kinerja akhirnya mendatar dengan peningkatan jumlah parameter, kinerja penalaran secara konsisten meningkat dengan lebih banyak waktu yang dihabiskan untuk _berpikir_ selama inferensi. Beberapa studi menunjukkan bahwa membiarkan model lebih banyak waktu untuk menyelesaikan masalah kompleks langkah demi langkah dapat meningkatkan kemampuan pemecahan masalah mereka di domain tertentu. Pendekatan ini, kadang disebut _penskalaan waktu inferensi (inference-time scaling)_, masih merupakan area penelitian yang berkembang dengan hasil awal yang menjanjikan.

Dinamika penskalaan yang muncul ini menunjukkan bahwa sementara penskalaan pra-pelatihan mungkin mendekati hasil yang semakin berkurang (diminishing returns), penskalaan pasca-pelatihan dan waktu inferensi mewakili batas baru yang menjanjikan. Hubungan antara hukum penskalaan ini dan kemampuan mengikuti instruksi (instruction-following) sangat menonjol—model harus memiliki kemampuan mengikuti instruksi yang cukup kuat untuk menunjukkan manfaat penskalaan waktu uji ini. Ini menciptakan kasus yang menarik untuk memusatkan upaya penelitian pada peningkatan penalaran waktu inferensi daripada hanya memperluas ukuran model.

Setelah memeriksa keterbatasan teknis penskalaan dan alternatif yang muncul, kita sekarang beralih ke konsekuensi ekonomi dari perkembangan ini. Seperti yang akan kita lihat, pergeseran dari penskalaan murni ke pendekatan yang lebih efisien memiliki implikasi signifikan untuk dinamika pasar, pola investasi, dan peluang penciptaan nilai.

## Transformasi ekonomi dan industri

Mengintegrasikan AI generatif menjanjikan peningkatan produktivitas yang sangat besar melalui mengotomatisasi tugas di berbagai sektor, sementara berpotensi menyebabkan gangguan tenaga kerja karena kecepatan perubahan. Menurut _Indeks Dampak Kecerdasan Buatan Global_ PwC 2023 dan laporan JPMorgan 2024 _Dampak Ekonomi AI Generatif_, AI dapat berkontribusi hingga $15,7 triliun untuk ekonomi global pada 2030, meningkatkan PDB global hingga 14%. Dampak ekonomi ini akan didistribusikan secara tidak merata, dengan China berpotensi melihat peningkatan PDB 26% dan Amerika Utara sekitar 14%. Sektor yang diharapkan melihat dampak tertinggi termasuk (secara berurutan):

- Kesehatan (Healthcare)
- Otomotif (Automotive)
- Layanan keuangan (Financial services)
- Transportasi dan logistik (Transportation and logistics)

Laporan JPM menyoroti bahwa AI lebih dari sekadar otomatisasi sederhana—ia secara fundamental meningkatkan kemampuan bisnis. Keuntungan masa depan kemungkinan akan menyebar di seluruh ekonomi seiring kepemimpinan sektor teknologi berkembang dan inovasi menyebar di berbagai industri.

Evolusi adopsi AI dapat lebih dipahami dalam konteks revolusi teknologi sebelumnya, yang biasanya mengikuti pola kurva-S dengan tiga fase berbeda, seperti yang dijelaskan dalam karya seminal Everett Rogers _Difusi Inovasi (Diffusion of Innovations)_. Sementara revolusi teknologi tipikal secara historis mengikuti fase ini selama beberapa dekade, _Situational Awareness: The Decade Ahead_ (2024) Leopold Aschenbrenner berpendapat bahwa implementasi AI mungkin mengikuti garis waktu yang terkompresi karena kemampuan uniknya untuk meningkatkan diri sendiri dan mempercepat pengembangannya sendiri. Analisis Aschenbrenner menunjukkan bahwa kurva-S tradisional mungkin sangat curam untuk teknologi AI, berpotensi mengompresi siklus adopsi yang sebelumnya membutuhkan dekade menjadi tahun:

1. **Fase pembelajaran (Learning phase) (5-30 tahun)**: Eksperimen awal dan pengembangan infrastruktur
2. **Fase pelaksanaan (Doing phase) (10-20 tahun)**: Penskalaan cepat setelah infrastruktur yang memungkinkan matang
3. **Fase optimalisasi (Optimization phase) (berkelanjutan)**: Peningkatan bertahap setelah saturasi

Analisis terbaru menunjukkan bahwa implementasi AI kemungkinan akan mengikuti lintasan bertahap yang lebih kompleks:

- **2030-2040**: Manufaktur, logistik, dan tugas kantor berulang dapat mencapai 70-90% otomatisasi
- **2040-2050**: Sektor jasa seperti kesehatan dan pendidikan mungkin mencapai 40-60% otomatisasi karena robot humanoid dan kemampuan AGI matang
- **Pasca-2050**: Pertimbangan sosial dan etika dapat menunda otomatisasi penuh peran yang memerlukan empati

Berdasarkan analisis dari "Laporan Masa Depan Pekerjaan 2023" Forum Ekonomi Dunia dan penelitian McKinsey Global Institute tentang potensi otomatisasi di berbagai sektor, kita dapat memetakan potensi otomatisasi relatif di industri kunci:

Tingkat otomatisasi spesifik dan proyeksi mengungkapkan tingkat adopsi yang bervariasi:

| Sektor                                  | Potensi Otomatisasi                                                     | Penggerak Utama                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Manufaktur                              | Tinggi—terutama dalam tugas berulang dan lingkungan terstruktur         | Robot kolaboratif (collaborative robots), penglihatan mesin (machine vision), kontrol kualitas AI |
| Logistik/Gudang (Logistics/Warehousing) | Tinggi—khususnya dalam penyortiran, pemilihan (picking), dan inventaris | Robot bergerak otonom (autonomous mobile robots) (AMRs), sistem penyortiran otomatis              |
| Kesehatan (Healthcare)                  | Sedang—terkonsentrasi dalam tugas administratif dan diagnostik          | Bantuan diagnostik AI, operasi robotik, dokumentasi otomatis                                      |
| Ritel (Retail)                          | Sedang—terutama dalam proses inventaris dan checkout                    | Checkout mandiri (self-checkout), manajemen inventaris, pemenuhan otomatis                        |

Tabel 10.2: Keadaan tingkat otomatisasi spesifik sektor dan proyeksi

Data ini mendukung pandangan bernuansa tentang garis waktu otomatisasi di berbagai sektor. Sementara manufaktur dan _logistik_ berkembang pesat menuju tingkat otomatisasi tinggi, sektor jasa dengan interaksi manusia kompleks menghadapi hambatan yang lebih signifikan.

Perkiraan McKinsey sebelumnya dari 2023 menunjukkan bahwa LLM dapat secara langsung mengotomatisasi 20% tugas dan secara tidak langsung mengubah 50% tugas. Namun, implementasi terbukti lebih menantang dari yang diantisipasi. Penyebaran paling sukses adalah yang meningkatkan kemampuan manusia daripada mencoba penggantian penuh.

### Transformasi spesifik industri dan dinamika kompetitif

Lanskap kompetitif untuk penyedia AI telah berkembang secara signifikan pada 2024-2025. Kompetisi harga telah meningkat seiring konvergensi kemampuan teknis di seluruh vendor, memberikan tekanan pada margin keuntungan di seluruh industri. Perusahaan menghadapi tantangan dalam membangun keunggulan kompetitif berkelanjutan di luar teknologi inti mereka, karena diferensiasi semakin bergantung pada keahlian domain, integrasi solusi, dan kualitas layanan daripada kinerja model mentah. Tingkat adopsi perusahaan tetap sederhana dibandingkan dengan proyeksi awal, menunjukkan bahwa investasi infrastruktur masif yang dibuat di bawah hipotesis penskalaan mungkin berjuang untuk menghasilkan pengembalian yang memadai dalam jangka pendek.

Pengadopsi manufaktur terkemuka—seperti Pabrik Mercusuar Global (Global Lighthouse factories)—sudah mengotomatisasi 50-80% tugas menggunakan robotika bertenaga AI, mencapai ROI dalam 2-3 tahun. Menurut Analisis Pasar Robot Kolaboratif ABI Research 2023 ([https://www.abiresearch.com/press/collaborative-robots-pioneer-automation-revolution-market-to-reach-us7.2-billion-by-2030](https://www.abiresearch.com/press/collaborative-robots-pioneer-automation-revolution-market-to-reach-us7.2-billion-by-2030)), robot kolaboratif mengalami waktu penyebaran yang lebih cepat daripada robot industri tradisional, dengan periode implementasi rata-rata 30-40% lebih pendek. Namun, kemajuan ini tetap terutama efektif di lingkungan terstruktur. Kesenjangan antara fasilitas perintis dan rata-rata industri (saat ini pada 45-50% otomatisasi) mengilustrasikan potensi dan tantangan implementasi di depan.

Di industri kreatif, kita melihat kemajuan dalam domain spesifik. Alat pengembangan perangkat lunak seperti GitHub Copilot mengubah cara kerja pengembang, meskipun persentase spesifik otomatisasi tugas tetap sulit untuk diukur secara tepat. Demikian pula, alat analisis data semakin menangani tugas rutin di seluruh keuangan dan pemasaran, meskipun tingkat pastinya sangat bervariasi berdasarkan implementasi. Menurut penelitian McKinsey Global Institute 2017, hanya sekitar 5% pekerjaan yang dapat sepenuhnya diotomatisasi oleh teknologi yang didemonstrasikan, sementara banyak lagi yang memiliki bagian kegiatan yang dapat diotomatisasi signifikan (sekitar 30% kegiatan dapat diotomatisasi di 60% pekerjaan). Ini menunjukkan bahwa sebagian besar implementasi sukses adalah meningkatkan daripada sepenuhnya menggantikan kemampuan manusia.

### Evolusi pekerjaan dan implikasi keterampilan

Saat adopsi otomatisasi berkembang di berbagai industri, dampak pada pekerjaan akan sangat bervariasi berdasarkan sektor dan garis waktu. Berdasarkan tingkat adopsi saat ini dan proyeksi, kita dapat mengantisipasi bagaimana peran spesifik akan berkembang.

#### Dampak jangka pendek (2025-2035)

Saat adopsi otomatisasi berkembang di berbagai industri, dampak pada pekerjaan akan sangat bervariasi berdasarkan sektor dan garis waktu. Sementara persentase otomatisasi tepat sulit diprediksi, kita dapat mengidentifikasi pola jelas dalam bagaimana peran spesifik kemungkinan berkembang.

Menurut penelitian McKinsey Global Institute, hanya sekitar 5% pekerjaan yang dapat sepenuhnya diotomatisasi dengan teknologi saat ini, meskipun sekitar 60% pekerjaan memiliki setidaknya 30% dari kegiatan penyusunnya yang dapat diotomatisasi. Ini menunjukkan bahwa transformasi pekerjaan—daripada penggantian besar-besaran—akan menjadi pola dominan seiring kemampuan AI maju. Implementasi paling sukses hingga saat ini telah meningkatkan kemampuan manusia daripada sepenuhnya menggantikan pekerja.

Potensi otomatisasi sangat bervariasi di berbagai sektor. Manufaktur dan logistik, dengan lingkungan terstruktur dan tugas berulang, menunjukkan potensi otomatisasi yang lebih tinggi daripada sektor yang memerlukan interaksi manusia kompleks seperti kesehatan dan pendidikan. Diferensial ini menciptakan garis waktu tidak merata untuk transformasi di seluruh ekonomi.

#### Dampak menengah (2035-2045)

Saat sektor jasa mencapai tingkat otomatisasi 40-60% dalam dekade berikutnya, kita dapat mengharapkan transformasi signifikan dalam peran profesional tradisional:

- **Profesi hukum (Legal profession)**: Pekerjaan hukum rutin seperti tinjauan dokumen dan persiapan draf akan sebagian besar diotomatisasi, secara fundamental mengubah peran pekerjaan untuk pengacara yunior dan paralegal. Firma hukum yang telah memulai transisi ini melaporkan mempertahankan jumlah karyawan sambil secara signifikan meningkatkan kapasitas kasus.
- **Pendidikan (Education)**: Guru akan menggunakan AI untuk persiapan kursus, tugas administratif, dan dukungan siswa yang dipersonalisasi. Siswa sudah menggunakan AI generatif untuk mempelajari konsep baru melalui interaksi pengajaran yang dipersonalisasi, mengajukan pertanyaan lanjutan untuk memperjelas pemahaman dengan kecepatan mereka sendiri. Peran guru akan berkembang menuju bimbingan (mentorship), pengembangan pemikiran kritis, dan desain pembelajaran kreatif daripada pengiriman informasi murni, berfokus pada aspek di mana bimbingan manusia menambah nilai paling banyak.
- **Kesehatan (Healthcare)**: Sementara pengambilan keputusan klinis akan tetap terutama manusia, dukungan diagnostik, dokumentasi, dan pemantauan rutin akan semakin diotomatisasi, memungkinkan penyedia layanan kesehatan fokus pada kasus kompleks dan hubungan pasien.

#### Pergeseran jangka panjang (2045 dan seterusnya)

Saat teknologi mendekati peran yang memerlukan lebih banyak empati, kita dapat mengharapkan yang berikut diminati:

- **Keahlian khusus (Specialized expertise)**: Permintaan akan tumbuh signifikan untuk ahli dalam etika AI, regulasi, pengawasan keamanan, dan desain kolaborasi manusia-AI. Peran ini akan sangat penting untuk memastikan hasil yang bertanggung jawab seiring sistem menjadi lebih otonom.
- **Bidang kreatif (Creative fields)**: Musisi dan seniman akan mengembangkan bentuk baru kolaborasi manusia-AI, berpotensi meningkatkan ekspresi kreatif dan aksesibilitas sambil menimbulkan pertanyaan baru tentang atribusi dan orisinalitas.
- **Kepemimpinan dan strategi (Leadership and strategy)**: Peran yang memerlukan penilaian kompleks, penalaran etika, dan manajemen pemangku kepentingan akan menjadi salah satu yang terakhir melihat otomatisasi signifikan, berpotensi meningkatkan nilai relatif mereka dalam ekonomi.

### Distribusi ekonomi dan pertimbangan kesetaraan (equity)

Tanpa intervensi kebijakan yang disengaja, manfaat ekonomi AI mungkin terkumpul secara tidak proporsional kepada mereka yang memiliki modal, keterampilan, dan infrastruktur untuk memanfaatkan teknologi ini, berpotensi memperlebar ketidaksetaraan yang ada. Kekhawatiran ini sangat relevan untuk:

- **Ketimpangan geografis (Geographic disparities)**: Wilayah dengan infrastruktur teknologi dan sistem pendidikan yang kuat mungkin semakin unggul dari daerah kurang berkembang.
- **Ketidaksetaraan berbasis keterampilan (Skills-based inequality)**: Pekerja dengan pendidikan dan kemampuan beradaptasi untuk melengkapi sistem AI kemungkinan akan melihat pertumbuhan upah, sementara yang lain mungkin menghadapi penggantian atau stagnasi upah.
- **Konsentrasi modal (Capital concentration)**: Organisasi yang berhasil mengimplementasikan AI dapat merebut pangsa pasar yang tidak proporsional, berpotensi menyebabkan konsentrasi industri yang lebih besar.

Mengatasi tantangan ini akan memerlukan pendekatan kebijakan yang terkoordinasi:

- Investasi dalam program pendidikan dan pelatihan ulang untuk membantu pekerja beradaptasi dengan perubahan persyaratan pekerjaan
- Kerangka kerja regulasi yang mempromosikan persaingan dan mencegah konsentrasi pasar yang berlebihan
- Dukungan yang ditargetkan untuk wilayah dan komunitas yang menghadapi gangguan signifikan

Pola konsisten di semua kerangka waktu adalah bahwa sementara tugas rutin menghadapi otomatisasi yang meningkat (pada tingkat yang ditentukan oleh faktor spesifik sektor), keahlian manusia untuk memandu sistem AI dan memastikan hasil yang bertanggung jawab tetap penting. Evolusi ini menunjukkan kita harus mengharapkan transformasi daripada penggantian besar-besaran, dengan ahli teknis tetap menjadi kunci untuk mengembangkan alat AI dan mewujudkan potensi bisnisnya.

Dengan mengotomatisasi tugas rutin, model AI lanjutan pada akhirnya dapat membebaskan waktu manusia untuk pekerjaan bernilai lebih tinggi, berpotensi meningkatkan output ekonomi keseluruhan sambil menciptakan tantangan transisi yang memerlukan respons kebijakan yang bijaksana. Pengembangan AI yang mampu bernalar kemungkinan akan mempercepat transformasi ini dalam peran analitis, sementara memiliki dampak kurang langsung pada peran yang memerlukan kecerdasan emosional dan keterampilan interpersonal.

## Implikasi sosial

Sebagai pengembang dan pemangku kepentingan dalam ekosistem AI, memahami implikasi sosial yang lebih luas dari teknologi ini bukan hanya latihan teoritis tetapi kebutuhan praktis. Keputusan teknis yang kita buat hari ini akan membentuk dampak AI pada lingkungan informasi, sistem kekayaan intelektual, pola ketenagakerjaan, dan lanskap regulasi besok. Dengan memeriksa dimensi sosial ini, pembaca dapat lebih mengantisipasi tantangan, merancang sistem yang lebih bertanggung jawab, dan berkontribusi membentuk masa depan di mana AI generatif menciptakan manfaat luas sambil meminimalkan potensi bahaya. Selain itu, kesadaran akan implikasi ini membantu menavigasi pertimbangan etika dan regulasi kompleks yang semakin memengaruhi pengembangan dan penyebaran AI.

### Misinformasi dan keamanan siber (cybersecurity)

AI menghadirkan pedang bermata dua untuk integritas informasi dan keamanan. Sementara itu memungkinkan deteksi informasi palsu yang lebih baik, itu secara bersamaan memfasilitasi penciptaan misinformasi yang semakin canggih dalam skala dan personalisasi yang belum pernah terjadi sebelumnya. AI generatif dapat membuat kampanye disinformasi yang ditargetkan yang disesuaikan dengan demografi dan individu tertentu, membuat orang lebih sulit membedakan antara konten otentik dan yang dimanipulasi. Saat dikombinasikan dengan kemampuan mikro-targeting (micro-targeting), ini memungkinkan manipulasi opini publik yang presisi di seluruh platform sosial.

Di luar misinformasi murni, AI generatif mempercepat serangan rekayasa sosial (social engineering attacks) dengan memungkinkan pesan phishing yang dipersonalisasi yang meniru gaya penulisan kontak tepercaya. Ini juga dapat menghasilkan kode untuk perangkat lunak berbahaya (malware), membuat serangan canggih dapat diakses oleh pelaku ancaman yang kurang terampil secara teknis.

Fenomena deepfake mewakili perkembangan yang paling mengkhawatirkan. Sistem AI sekarang dapat menghasilkan video, gambar, dan audio palsu yang realistis yang tampaknya menunjukkan orang nyata mengatakan atau melakukan hal-hal yang tidak pernah mereka lakukan. Teknologi ini mengancam untuk mengikis kepercayaan pada media dan institusi sambil memberikan penyangkalan yang masuk akal untuk kesalahan yang sebenarnya ("itu hanya palsu AI").

Asimetri antara penciptaan dan deteksi menimbulkan tantangan signifikan—secara umum lebih mudah dan lebih murah untuk menghasilkan konten palsu yang meyakinkan daripada membangun sistem untuk mendeteksinya. Ini menciptakan keunggulan persisten bagi mereka yang menyebarkan misinformasi.

Keterbatasan dalam pendekatan penskalaan memiliki implikasi penting untuk kekhawatiran misinformasi. Sementara model yang lebih kuat diharapkan mengembangkan landasan faktual dan kemampuan penalaran yang lebih baik, halusinasi yang persisten bahkan dalam sistem paling canggih menunjukkan bahwa solusi teknis saja mungkin tidak cukup. Ini telah menggeser fokus ke pendekatan hibrida yang menggabungkan AI dengan pengawasan manusia dan verifikasi pengetahuan eksternal.

Untuk mengatasi ancaman ini, beberapa pendekatan pelengkap diperlukan:

- **Pengamanan teknis (Technical safeguards)**: Sistem asal usul konten (content provenance systems), penandaan air digital (digital watermarking), dan algoritma deteksi lanjutan
- **Literasi media (Media literacy)**: Pendidikan luas tentang mengidentifikasi konten yang dimanipulasi dan mengevaluasi sumber informasi
- **Kerangka kerja regulasi (Regulatory frameworks)**: Hukum yang menangani deepfake dan disinformasi otomatis
- **Tanggung jawab platform (Platform responsibility)**: Moderasi konten dan sistem autentikasi yang ditingkatkan
- **Jaringan deteksi kolaboratif (Collaborative detection networks)**: Berbagi pola disinformasi lintas platform

Kombinasi kemampuan generatif AI dengan mekanisme distribusi skala internet menghadirkan tantangan yang belum pernah terjadi sebelumnya pada ekosistem informasi yang mendasari masyarakat demokratis. Mengatasi ini akan memerlukan upaya terkoordinasi di seluruh domain teknis, pendidikan, dan kebijakan.

### Tantangan hak cipta (copyright) dan atribusi

AI generatif menimbulkan pertanyaan hak cipta penting untuk pengembang. Putusan pengadilan baru-baru ini ([https://www.reuters.com/world/us/us-appeals-court-rejects-copyrights-ai-generated-art-lacking-human-creator-2025-03-18/](https://www.reuters.com/world/us/us-appeals-court-rejects-copyrights-ai-generated-art-lacking-human-creator-2025-03-18/)) telah menetapkan bahwa konten yang dihasilkan AI tanpa masukan kreatif manusia yang signifikan tidak dapat menerima perlindungan hak cipta. Pengadilan Banding AS secara definitif memutuskan pada Maret 2025 bahwa "kepengarangan manusia diperlukan untuk pendaftaran" di bawah hukum hak cipta, mengonfirmasi karya yang dibuat semata-mata oleh AI tidak dapat diberi hak cipta.

Pertanyaan kepemilikan tergantung pada keterlibatan manusia. Keluaran hanya-AI tetap tidak dapat diberi hak cipta, sementara keluaran AI yang diarahkan manusia dengan pilihan kreatif mungkin dapat diberi hak cipta, dan kreasi manusia yang dibantu AI mempertahankan perlindungan hak cipta standar.

Pertanyaan melatih LLM pada karya berhak cipta tetap diperdebatkan. Sementara beberapa menegaskan ini merupakan penggunaan wajar (fair use) sebagai proses transformatif, kasus terbaru telah menantang posisi ini. Putusan Thomson Reuters Februari 2025 ([https://www.lexology.com/library/detail.aspx?g=8528c643-bc11-4e1d-b4ab-b467cd641e4c](https://www.lexology.com/library/detail.aspx?g=8528c643-bc11-4e1d-b4ab-b467cd641e4c)) menolak pembelaan penggunaan wajar untuk AI yang dilatih pada materi hukum berhak cipta.

Masalah ini secara signifikan memengaruhi industri kreatif di mana model kompensasi yang mapan bergantung pada kepemilikan dan atribusi yang jelas. Tantangan ini sangat akut dalam seni visual, musik, dan sastra, di mana AI generatif dapat menghasilkan karya yang secara stilistik mirip dengan seniman atau penulis tertentu.

Solusi yang diusulkan termasuk sistem asal usul konten yang melacak sumber pelatihan, model kompensasi yang mendistribusikan royalti kepada pencipta yang karyanya menginformasikan AI, penandaan air teknis untuk membedakan konten yang dihasilkan AI, dan kerangka kerja hukum yang menetapkan standar atribusi yang jelas.

Saat mengimplementasikan aplikasi LangChain, pengembang harus melacak dan mengaitkan konten sumber, mengimplementasikan filter untuk mencegah reproduksi verbatim, mendokumentasikan sumber data yang digunakan dalam penyempurnaan, dan mempertimbangkan pendekatan yang ditingkatkan pengambilan (retrieval-augmented) yang mengutip sumber dengan benar.

Kerangka kerja internasional bervariasi, dengan Undang-Undang AI Uni Eropa 2024 menetapkan pengecualian penambangan data (data mining) spesifik dengan hak opt-out pemegang hak cipta mulai Agustus 2025. Dilema ini menyoroti kebutuhan mendesak untuk kerangka kerja hukum yang dapat mengikuti kemajuan teknologi dan menavigasi interaksi kompleks antara pemegang hak dan konten yang dihasilkan AI. Saat standar hukum berkembang, sistem fleksibel yang dapat beradaptasi dengan perubahan persyaratan menawarkan perlindungan terbaik untuk pengembang dan pengguna.

### Regulasi dan tantangan implementasi

Mewujudkan potensi AI generatif dengan cara yang bertanggung jawab melibatkan mengatasi masalah hukum, etika, dan regulasi. Undang-Undang AI Uni Eropa mengambil pendekatan komprehensif berbasis risiko untuk mengatur sistem AI. Ini mengategorikan sistem AI berdasarkan tingkat risiko:

- **Risiko minimal (Minimal risk)**: Aplikasi AI dasar dengan potensi terbatas untuk bahaya
- **Risiko terbatas (Limited risk)**: Sistem yang memerlukan kewajiban transparansi
- **Risiko tinggi (High risk)**: Aplikasi dalam infrastruktur kritis, pendidikan, ketenagakerjaan, dan layanan esensial
- **Risiko tidak dapat diterima (Unacceptable risk)**: Sistem yang dianggap menimbulkan ancaman mendasar terhadap hak dan keselamatan

Aplikasi AI berisiko tinggi seperti perangkat lunak medis dan alat rekrutmen menghadapi persyaratan ketat mengenai kualitas data, transparansi, pengawasan manusia, dan mitigasi risiko. Hukum secara eksplisit melarang penggunaan AI tertentu yang dianggap menimbulkan "risiko tidak dapat diterima" terhadap hak fundamental, seperti sistem penilaian sosial (social scoring systems) dan praktik manipulatif yang menargetkan kelompok rentan. Undang-Undang AI juga memaksakan kewajiban transparansi pada pengembang dan mencakup aturan spesifik untuk model AI tujuan umum dengan potensi dampak tinggi.

Ada juga permintaan yang tumbuh untuk transparansi algoritmik (algorithmic transparency), dengan perusahaan teknologi dan pengembang menghadapi tekanan untuk mengungkapkan lebih banyak tentang cara kerja internal sistem mereka. Namun, perusahaan sering menolak pengungkapan, berargumen bahwa mengungkapkan informasi kepemilikan akan merugikan keunggulan kompetitif mereka. Ketegangan antara transparansi dan perlindungan kekayaan intelektual ini tetap belum terselesaikan, dengan model open-source berpotensi mendorong transparansi yang lebih besar sementara sistem berpemilik mempertahankan lebih banyak ketidakjelasan.

Pendekatan saat ini untuk moderasi konten, seperti Undang-Undang Penegakan Jaringan Jerman (German Network Enforcement Act) (NetzDG), yang memaksakan kerangka waktu 24 jam untuk platform menghapus berita palsu dan ujaran kebencian, telah terbukti tidak praktis.

Pengakuan keterbatasan penskalaan memiliki implikasi penting untuk regulasi. Pendekatan awal untuk tata kelola AI berfokus berat pada mengatur akses ke sumber daya komputasi. Namun, inovasi terbaru menunjukkan bahwa kemampuan terkini dapat dicapai dengan komputasi yang jauh lebih sedikit. Ini telah mendorong pergeseran dalam kerangka kerja regulasi ke arah mengatur kemampuan dan aplikasi AI daripada sumber daya yang digunakan untuk melatihnya.

Untuk memaksimalkan manfaat sambil mengurangi risiko, organisasi harus memastikan pengawasan manusia, keberagaman, dan transparansi dalam pengembangan AI. Memasukkan pelatihan etika ke dalam kurikulum ilmu komputer dapat membantu mengurangi bias dalam kode AI dengan mengajarkan pengembang cara membangun aplikasi yang etis dari desain (ethical by design). Pembuat kebijakan, di sisi lain, mungkin perlu mengimplementasikan pengaman (guardrails) mencegah penyalahgunaan sambil memberikan pekerja dukungan untuk transisi seiring kegiatan bergeser.

## Ringkasan

Saat kami menyimpulkan eksplorasi AI generatif dengan LangChain ini, kami harap Anda dilengkapi tidak hanya dengan pengetahuan teknis tetapi dengan pemahaman yang lebih dalam tentang ke mana teknologi ini menuju. Perjalanan dari aplikasi LLM dasar ke sistem agenik yang canggih mewakili salah satu batas paling menarik dalam komputasi saat ini.

Implementasi praktis yang telah kami bahas di seluruh buku ini—dari RAG ke sistem multi-agen, dari agen pengembangan perangkat lunak ke strategi penyebaran produksi—memberikan fondasi untuk membangun aplikasi AI yang kuat dan bertanggung jawab hari ini. Namun seperti yang telah kita lihat di bab terakhir ini, bidang ini terus berkembang dengan cepat melampaui pendekatan penskalaan sederhana menuju paradigma yang lebih efisien, khusus, dan terdistribusi.

Kami mendorong Anda untuk menerapkan apa yang telah Anda pelajari, untuk bereksperimen dengan teknik yang telah kami eksplorasi, dan untuk berkontribusi pada ekosistem yang berkembang ini. Repositori yang terkait dengan buku ini ([https://github.com/benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain)) akan dipelihara dan diperbarui seiring LangChain dan lanskap AI generatif yang lebih luas terus berkembang.

Masa depan teknologi ini akan dibentuk oleh praktisi yang membangun dengannya. Dengan mengembangkan implementasi yang bijaksana, efektif, dan bertanggung jawab, Anda dapat membantu memastikan bahwa AI generatif memenuhi janjinya sebagai teknologi transformatif yang meningkatkan kemampuan manusia dan membawa tantangan yang bermakna.

Kami bersemangat untuk melihat apa yang Anda bangun!

## Berlangganan buletin mingguan kami

Berlangganan AI_Distilled, buletin utama untuk profesional, peneliti, dan inovator AI, di [https://packt.link/Q5UyU](https://packt.link/Q5UyU).

![Kode QR buletin](Images/Newsletter_QRcode1.jpg)
