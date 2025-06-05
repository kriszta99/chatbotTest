from flask import Flask, jsonify, render_template, request, redirect
from upstash_vector import Index
import numpy as np
from collections import defaultdict
import os
import pandas as pd
import time
import openai
from bert_score import score
from google import genai
from upstash_vector.types import SparseVector, QueryMode
from concurrent.futures import ThreadPoolExecutor, wait
import threading
from dotenv import load_dotenv

# Környezeti változók betöltése
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
api_key = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
vector_db = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
loading_done = False
loading_started = False
loading_lock = threading.Lock()


results_ = []
#lekérem az összes vektort (parhuzamosan hogy gyorsitsam a lekeresi időt) egy globalis listaba az adatbazisból ösz:1158
def load_all_vectors_to_list():
    global results_, loading_done
    start_total_load = time.time()
    results_.clear()
    loading_done = False

    def fetch_range(cursor, limit):
        return vector_db.range(cursor=cursor, limit=limit, prefix="chunk_", include_metadata=True).vectors

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fetch_range, "0", 386),
            executor.submit(fetch_range, "386", 386),
            executor.submit(fetch_range, "772", 386)
        ]

        for future in futures:
            results_.extend(future.result())

    end_total_load = time.time()

    print(f"Betöltve {len(results_)} vektor, idő: {end_total_load - start_total_load:.2f} mp")
    loading_done = True




# OpenAI embedding-é alakitom a felhasználó kérdését
"""def get_embedding(text: str, model="text-embedding-ada-002") -> list:
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding"""
    
def get_embedding(text: str, model="text-embedding-ada-002") -> np.ndarray:
    response = openai.embeddings.create(input=text, model=model)
    # Az új API-ban a válasz egy objektum, nem közvetlenül szótár
    embedding = response.data[0].embedding
    return np.array(embedding)

def get_chunk_id_from_embedding(query_embedding):
    #legközelebbi vektorok lekérdezése
    results = vector_db.query(vector=query_embedding, include_metadata=True,top_k=50,query_mode=QueryMode.DENSE)    
    chunk_ids = [result.metadata.get('chunk_id') for result in results]
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    chunk_ids = []
    for result in sorted_results:
        if result.score >= 0.85:
            #print(f"chunk_id: {result.metadata.get('chunk_id')}\nText: {result.metadata.get('text')}\nScore: {result.score}\n\n")
            chunk_ids.append(result.metadata.get('chunk_id'))

    print(len(chunk_ids)) 
    return chunk_ids


def query_by_chunk_id(chunk_ids): 
    # Szűrés a chunk_id alapján
    filtered_results = [
        r for r in results_
        if r.metadata.get("chunk_id") in chunk_ids
    ]
  
    if not filtered_results:
        print("\nNincs találat.")
        return {}

     #csoportosítom chunk_id szerint a kontextus megmaradása érdekében
    grouped_by_chunk_id = defaultdict(list)
    for r in filtered_results:
        cid = r.metadata.get("chunk_id")
        grouped_by_chunk_id[cid].append(r)

    #rendezem minden csoporton belül chunk_order szerint hogy a kontextusban a megfeleő sorendben legyenek összeallitva a szövegek
    for cid in grouped_by_chunk_id:
        grouped_by_chunk_id[cid].sort(key=lambda r: r.metadata.get("chunk_order", 0))

    return grouped_by_chunk_id

def get_context_text(query_embedding):
    #chunk_ids = get_chunk_id_from_embedding(query_embedding,user_question)
    chunk_ids = get_chunk_id_from_embedding(query_embedding)
    if not chunk_ids:
        return "Nem található chunk_id."

    grouped_results = query_by_chunk_id(chunk_ids)
    if not grouped_results:
        return f"Nincs találat a(z) {chunk_ids} chunk_id-re."

    # minden chunk_id-hoz tartozó szöveg összefűzése, chunk_order szerint
    context_parts = []
    for chunk_id in sorted(grouped_results.keys()):
        texts = [r.metadata.get('text','') for r in grouped_results[chunk_id]]
        text_joined = "\n".join(texts)
        context_parts.append(f"{chunk_id}\n{text_joined}")

    # chunk_id csoportokat elválasztjuk dupla sortöréssel
    context = "\n\n".join(context_parts)
    return context



#az LLM model segitségével választ generalok a feltett kérdésre
def get_llm_response(context, question):
    client = genai.Client(api_key=api_key)

    full_prompt = f"<context>{context}</context>Kérem, válaszoljon az alábbi kérdésre a fent megadott kontextus alapján:<user_query>{question}</user_query>\nVálasz:"


    response = client.models.generate_content(
        #model="gemini-2.0-flash-thinking-exp-01-21",
        #model = "gemini-2.0-flash",
        #model="gemini-1.5-flash",
        model="gemini-2.0-flash",


        contents=[full_prompt]
    )
    return response.text

def save_timings_to_excel(filepath, question, t_embed, t_ctx, t_llm, t_total,bertscore_f1):
    new_row = {
        "felhasznaló kérdése": question,
        "embedding generálásai idő": round(t_embed, 3),
        "kontextus összeállitási idő": round(t_ctx, 3),
        "LLM feldolgozási idő": round(t_llm, 3),
        "teljes feldoldozási idő": round(t_total, 3),
        "szemantikus hasonlóság méréke(BERTScore F1)": round(bertscore_f1,3)

    }


    # ha létezik a fájl beolvasom, ha nem létrehozom az új DataFrame-et
    if os.path.exists(filepath):
        df = pd.read_excel(filepath)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_excel(filepath, index=False)
    
ground_truths_vector_by_search_20 = [
    "2018-ban egy újabb szárny nyílt meg a Campuson, a C épület, amelyben két amfiteátrum, számos terem és laboratórium is létesült.",
    "dr. Horobeț Emil, egyetemi docens",
    """Vakációban (a nyári szünet alatt is):
    *   hétfő – péntek: 8:00 – 15:00
    *   szombat – vasárnap: zárva.""",
    "Ungvári Zsuzsi, PR-felelős",
    "Dr. Suba Réka - docens, tanszékvezető-helyettes, szakkoordinátor (Fordító és tolmács szak)",
    "A Mechatronika képzési szak 1990 után jelent meg a romániai felsőoktatásban, az ipar által támasztott igények kielégítésére, azon szakterületeken, ahol a szakember gépészmérnöki, villamosmérnöki és informatikai alapismeretekkel kell rendelkezzen.",
    "Dr. Márton Gyöngyvér, adjunktus, szakkoordinátor (Szoftverfejlesztés)",
    "helyben olvasás a 72 férőhelyes szabadpolcos olvasóteremben",
    """További nem kölcsönözhető dokumentumaink:
      *   tájékoztató segédkönyvek (szótárak, lexikonok),
      *   folyóiratok,
      *   szakdolgozatok (raktárból felkérhető).""",
    "Ha az Ön jövedelme csak fizetésből (egy vagy több) származik, akkor a **230**-as formanyomtatványt kell kitöltenie.",
    "Babos Annamária gazdasági igazgató",
    "* A Varjútábort, amit a már elballagott diákoknak szervezünk, hogy együtt tölthessünk egy hétvégét a régi idők emlékére.",
    """Feladatkör: Iroda felelős
    * Név: Ozsváth-Berényi Attila""",
    "A kar főépületét körülvevő 27 hektáros területen a kertészmérnöki szakos diákok gyakorlatoznak, akik parkosítják a terület egy részét.",
    "**Elnök:**\n* dr. Szabó László-Zsolt, egyetemi adjunktus",
    "Kölcsönözni csak személyesen, érvényes könyvtári igazolvánnyal lehet.",
    "A HÖK három legfontosabb feladata:\n* Információ továbbítás\n* Érdekképviselet\n* Rendezvényszervezés",
    "Vizsgaidőszakban szombat délelőttönként is nyitva tartunk, a pontos dátumokat előzőleg hirdetjük a Hírek, közlemények almenüben.",
    "Matematika-Informatika Tanszék (https://ms.sapientia.ro/hu/a-karrol/tanszeke/matematika-informatika-tanszek)",
    """## Marosvásárhelyi Kar
       Târgu-Mureş/Corunca (Marosvásárhely/Koronka), Calea Sighișoarei nr. 2.
       (26, 27, 44-es közszállítási vonal végállomása)
       Tel: +40 265 206 210
       fax: +40 265 206 211
       Postacím: 540485 Târgu-Mureş, O.p. 9, C.p. 4
       E-mail: office@ms.sapientia.ro
       Weboldal (http://ms.sapientia.ro/)"""
]

ground_truths_vector_by_search_40 = [
    "Akkreditációs felmérések dokumentumai (https://sapientia.ro/hu/az-egyetemrol/akkreditacio/akkreditacios-felmeresek-dokumentumai)",
    """Főbb tantárgyak:
        - algoritmusok és adatstruktúrák
        - programozási nyelvek
        - szoftvertechnológiák
        - mesterséges intelligencia
        - informatikai biztonság""",
    "4 év",
    """Fejlett mechatronikai rendszerek szak (https://ms.sapientia.ro/hu/felveteli/mesterkepzes/fejlett-mechatronikai-rendszerek-szak)
        Növényorvos szak (https://ms.sapientia.ro/hu/felveteli/mesterkepzes/novenyorvos-szak)
        Számítógépes irányítási rendszerek szak (https://ms.sapientia.ro/hu/felveteli/mesterkepzes/szamitogepes-iranyitasi-rendszerek-szak)
        Szoftverfejlesztés szak (https://ms.sapientia.ro/hu/felveteli/mesterkepzes/szoftverfejlesztes-szak)""",
    "15",
    "Alapképzés: Felvételi vizsga: 2025. július 15., kedd, 10 óra",
    "Mesterképzés: Online iratkozás: 2025. július 1 - július 15",
    "Felvételi szabályzatok (https://ms.sapientia.ro/hu/felveteli/felveteli-tudnivalok_/felveteli-szabalyzatok)",
    "Szükséges iratok (https://sapientia.ro/hu/felveteli/szukseges-iratok)",
    "Matematika 2023 (https://ms.sapientia.ro/content/docs/MS/Felveteli/2023/Felveteli_Matek2023SZEPT-B-Varians_1.pdf) (javítókulcs) (https://ms.sapientia.ro/content/docs/MS/Felveteli/2023/Felveteli_Matek2023SZEPT-B-VariansJAVITOKULCS_1.pdf)",
    "Lapozható tájékoztató füzet (https://issuu.com/sapientiaemte/docs/sap_felveteli-taj-2025_online)",
    "Gyakorlati órák pótlása (https://ms.sapientia.ro/hu/hallgatoknak/hallgatoi-tajekoztato/gyakorlati-orak-potlasa)",
    "Románia parlamentjének képviselőháza 2012. február 28-i ülésén megszavazta a Sapientia Erdélyi Magyar Tudományegyetem akkreditálását.",
    "Fő célja, hogy lehetővé tegye a munkaerőpiacon való elhelyezkedést biztosító diploma megszerzését, ezért az alapképzés során a gyakorlati képzés kap nagyobb hangsúlyt.",
    "Dr. Farkas Csaba, egyetemi tanár, a Sapientia EMTE általános rektorhelyettese",
    "- Informatika: dr. Jánosi-Rancz Tünde Katalin, adjunktus, tsuto@ms.sapientia.ro",
    "Az ösztöndíjat a hallgató egy összegben kapja meg a tanév elején az előző évi teljesítménye függvényében.",
    "Befizetési határidő: július 28.",
    "Jelenlegi összeg RON: 100 lej",
    "Adminisztratív díjak bankon keresztül utalhatóak a következő bankszámlára: RO48BTRLRONCRT0039221810 UNIVERSITATEA SAPIENTIA, CORUNCA, CF: RO14645945",
    "A felvételi jegy összetétele: 100% érettségi átlag",
    "Kari Erasmus Koordinátor: Biblia Csilla (sapierasmus@ms.sapientia.ro)",
    """Egyetem neve: Budapest University of Technology and Economics(Műszaki és Gazdaságtudományi Egyetem)
       Egyetem neve: Budapest University of Technology and Economics(Műszaki és Gazdaságtudományi Egyetem)
       Egyetem neve: University of Debrecen (Debreceni Egyetem)
       Egyetem neve: Eötvös Loránd University (Eötvös Loránd Tudományegyetem)
       Egyetem neve: Eszterházy Károly University(Eszterházy Károly Főiskola)
       Egyetem neve: Károli Gáspár University of The Refomed Church In Hungary (Károli Gáspár Református Egyetem)
       Egyetem neve: College of Nyíregyháza (Nyíregyházi Egyetem)
       Egyetem neve: University of Pannonia (Pannon Egyetem)
       Egyetem neve: University of Pécs (Pécsi Tudományegyetem)
       Egyetem neve: Semmelweis University (Semmelweis Egyetem)
       Egyetem neve: Szegedi Tudományegyetem
       Egyetem neve: Szent István Egyetem, Gödöllő
       Egyetem neve: Széchenyi István Egyetem, Győr""",
    "A Sapientia EMTE nemzetközi hallgatói útmutatója (https://ms.sapientia.ro/content/2011-2021/nemzetkozi-hallgatok/International-student-guide-Sapientia.pdf)",
    "Sapientia Erdélyi Magyar Tudományegyetem az oktatás megszervezésében az Európai Kreditátviteli Rendszert (ECTS) alkalmazza, amely által az egyetemi képzésben a végzettséget igazoló oklevél megszerzésének feltételéül előírt minden, tanulmányi munkaidő ráfordítással járó követelmény teljesítését kreditben méri.",
    "Vegyes intenzív programok (BIP)",
    "A Sapientia szó bölcsességet jelent, más szóval értő tanulmányozást és a tudásmegfontolt alkalmazását.",
    "A hátrányos helyzetű hallgatók hosszú mobilitás esetén havi 250 euró kiegészítő támogatást kapnak, úgy tanulmányi mobilitás, mind szakmai gyakorlat mobilitás esetén;",
    "A ***rövid vegyes mobilitás*** időtartama 5-30 nap, és ki kell egészülnie egy olyan **kötelező virtuális komponenssel**, amely lehetővé teszi az együttműködésen alapuló online tanulást és csapatmunkát – ösztöndíj azonban csak a fizikai mobilitási napokra jár.",
    "A Kiss Elemér Szakkollégium 2010. novemberében alakult a Sapientia Erdélyi Magyar Tudományegyetem Marosvásárhelyi Karán a MITIS Egyesület keretében.",
    "**Senior ösztöndíj** elnyerésére pályázatot nyújthatnak be az Akadémiának az Alapszabály 12. § (3) bekezdése szerinti Kárpát-medencei külső tagjai, valamint az Alapszabály 23. § (1) bekezdése szerinti Kárpát-medencei köztestületi külső tagok.",
    "Dr. Kelemen András egyetemi adjunktus - a kutatócsoport vezetője",
    "Fontosabb kutatási témáink: klaszterező és részlegesen felügyelt klaszterező algoritmusok, evolúciós algoritmusok, digitális jel- és képfeldolgozás, beszédtechnológia: folyamatos beszédfelismerés, beszédszintézis, nyelvi erőforrások fejlesztése, alkalmazások nagy adatbázisokkal, élettani rendszerek modellezése és szimulációja, protein együtthatási hálózatainak tanulmányozása, virtualizációs technikák.",
    """Bartha Zsolt, tanársegéd
       Vekov Géza Károly, tanársegéd
       Zsombori Gabriella, tanársegéd
       Garda-Mátyás Edit, tanársegéd""",
    "Távközlés (https://ms.sapientia.ro/content/docs/MS/TST-20230616T072159Z-001.zip)",
    "Oktatói versenyvizsgák (https://ms.sapientia.ro/hu/oktatas/oktatoi-versenyvizsgak)",
    """**Beiratkozáshoz szükséges iratok:**
    * **Beiratkozási kérvény** (alapképzés (https://ms.sapientia.ro/content/docs/MS/Zarovizsga/2025/cerere_alap_2025.docx) -, mesterképzés (https://ms.sapientia.ro/content/docs/MS/Zarovizsga/2025/cerere_mesteri_2025.docx) formanyomtatványa), amelyet a témavezető láttamoz;
    * Szakdolgozat/diplomaterv/disszertáció, elektronikusan PDF formátumban;
    * A témavezető/elővédési bizottság **referátuma**, amelynek tartalmaznia kell a dolgozat tartalmi értékelését és az ajánlott jegyet;
    * A *Turnitin*rendszerből generált **report**;
    * **2 db papír alapú**, útlevél típusú (3x4) **színes fénykép;**
    * A **szerző nyilatkozata** a dolgozat eredetiségéről;
    A dékáni hivatal az alábbi dokumentumokkal egészíti ki a végzettek beiratkozási iratcsomóját:
    * A **témavezető** **nyilatkozata** a vezetett dolgozat eredetiségére vonatkozóan;
    * A születési bizonyítvány **eredeti példánya** és a beiratkozás során **hitelesített másolata.**
    Az eredeti példányt a másolat helyszínen történő hitelesítése nyomán a jelentkező visszakapja.
    * **Érettségi oklevél eredetije**, valamint a **melléklet eredetije (törzskönyvi kivonat)** és a beiratkozás során hitelesített másolata, valamint esetenként a honosítási okirat;
    * **Egyetemi oklevél eredetije és a beiratkozás során hitelesített másolata** (mesteri szakok végzettjei esetében), valamint esetenként a **honosítási okirat**;
    * **Idegennyelvvizsga-bizonyítvány másolata.**
    Más nyelvvizsgaközpontok által kibocsátott bizonyítványok esetében a LinguaSap Központ általi elismerés is szükséges.""",
    "A nyelvvizsga két, egy írásbeli és egy szóbeli részből áll.",
    "A záróvizsga időpontja: 2025. július 7., 129-es terem",
    "A bentlakásban 3 szobatípus van: apartman, garzon, tetőtéri szoba."
]
@app.route("/init")
def init_load():
    global loading_started
    with loading_lock:
        if not loading_started:
            loading_started = True
            # Futtasd háttérszálban, hogy ne blokkolja a válaszadást
            threading.Thread(target=load_all_vectors_to_list).start()
        if not loading_done:
                return """
                <script>
                    alert("Kérlek várj, betöltés folyamatban...");
                    setTimeout(() => location.reload(), 2000);  // 2másodperc múlva újratölt
                </script>
                """
    
    return redirect("/chatbot")

# index, melyik ground truth-t hasonlítjuk össze az adott kérdésnél
current_gt_index = 0

@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    print(f"Process ID: {os.getpid()}")
    if request.method == 'POST':
        
        start_total = time.time()
        global current_gt_index

        data = request.get_json()
       
        user_question = data.get("question", "").strip()

        if not user_question:
            return jsonify({"error": "Nincs megadva kérdés"}), 400
        start_embed = time.time()
        embedding = get_embedding(user_question) #OpenAI API-val átalakítja a szöveget embedding-gé (vektorrá)
        end_embed = time.time()
        start_ctx = time.time()
        # kontextus összeállítása a lekérdezett embedding alapján
        context = get_context_text(embedding)
        end_ctx = time.time()
        start_llm = time.time()
        resp = get_llm_response(context,user_question)
        end_llm = time.time()
        end_total = time.time()

        if current_gt_index >= len(ground_truths_vector_by_search_40):
            return jsonify({"error": "Nincs több ground truth válasz teszteléshez."}), 400
        ground_truth = ground_truths_vector_by_search_40[current_gt_index]

        # pontosságot mérek BERTScore-dal
        P, R, F1 = score([resp], [ground_truth], lang="hu")
        bertscore_f1 = F1.mean().item()

        # növelem a ground truth indexet a következő kérdéshez
        current_gt_index += 1

        print(f"\n--- Időmérések (másodpercben) ---")
        t_embed = end_embed - start_embed
        print(f"Embedding generálás:     {t_embed:.3f}s")
        t_ctx = end_ctx - start_ctx
        print(f"Kontextus összeállítás:  {t_ctx:.3f}s")
        t_llm = end_llm - start_llm
        print(f"Gemini válasz:           {t_llm:.3f}s")
        t_total = end_total - start_total
        print(f"TELJES kérés feldolgozás: {t_total:.3f}s")
        print(f"Szemantikus hasonlóság méréke (bertscore_f1):{bertscore_f1}")
        #proba_valasz,p_context  = curent_context_from_context(embedding,user_question)
        #save_timings_to_excel("../vectorSearchTesting/timings_40_question_.xlsx", user_question, t_embed, t_ctx, t_llm, t_total,bertscore_f1)

        return jsonify({"answer": resp})
    

    return render_template('index.html')

if __name__ == '__main__':
    #load_all_vectors_to_list()
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=10000)

