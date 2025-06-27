from flask import Flask, jsonify, render_template, request, redirect
from upstash_vector import Index
import numpy as np
from collections import defaultdict
import os
import time
import requests
from openai import OpenAI
from google import genai
from upstash_vector.types import SparseVector, FusionAlgorithm, QueryMode
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

# betölttöm a környezeti változókat
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
api_key = os.getenv("GEMINI_API_KEY")
openRouter = os.getenv("OPEN_ROUTER_API_KEY")
api_key_pro = os.getenv("GEMINI_API_KEY_PRO")
deep_infra_api_key = os.getenv("DEEPINFRA_API_KEY")

app = Flask(__name__)

vector_db = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
loading_done = False
loading_started = False
loading_lock = threading.Lock()

results_ = []
#lekérem az összes vektort (parhuzamosan hogy gyorsitsam a lekeresi időt) egy globalis listaba az adatbazisból ösz:1159
def load_all_vectors_to_list():
    global results_, loading_done
    start_total_load = time.time()
    results_.clear()
    loading_done = False

    def fetch_range(cursor, limit):
        return vector_db.range(cursor=cursor, limit=limit, prefix="chunk_", include_metadata=True,include_vectors=False).vectors

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(fetch_range, "0", 579),
            executor.submit(fetch_range, "579", 580)
        ]

        for future in futures:
            results_.extend(future.result())

    end_total_load = time.time()
    #print(results_)
    print(f"Betöltve {len(results_)} vektor, össz idő: {end_total_load - start_total_load:.2f} s")
    loading_done = True

# OpenAI embedding-é alakitom a felhasználó kérdését    
def get_embedding(text: str, model="text-embedding-ada-002") -> np.ndarray:
    try:
        response = client.embeddings.create(input=text, model=model)
        # Az új API-ban a válasz egy objektum, nem közvetlenül szótár
        embedding = response.data[0].embedding
        return np.array(embedding)
    except Exception as e:
        if "503" in str(e) or "Rate limit" in str(e):
            raise RuntimeError(f"A text-embedding-ada-002 modell limitje elfogyott.")
        else:
            raise RuntimeError(f"Embedding model error: {str(e)}")

print("Warming up embedding model...")
get_embedding("warmup request", model="text-embedding-ada-002")
print("Embedding model ready.")

#bge-m3-multi sparse embedding modell segitségével atalakitom a felhasznaló kérdését Sparse vectorrá s visszatéritem 
def get_sparse_vector_from_query(user_query: str) -> SparseVector:
    url = "https://api.deepinfra.com/v1/inference/BAAI/bge-m3-multi"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deep_infra_api_key}"
    }
    data = {
        "inputs": [user_query],
        "dense": False,
        "sparse": True,
        "colbert": False
    }

    response = requests.post(url, headers=headers, json=data, timeout=(30))

    if response.status_code == 200:
        json_resp = response.json()
        # csak a sparse vektorokat szedjuk ki
        sparse_vec_full = json_resp['sparse'][0]

        arr = np.array(sparse_vec_full)
        # a nem nullás értékek kinyerése
        nonzero_indices = np.nonzero(arr)[0]
        nonzero_values = arr[nonzero_indices].astype(float)

        indices = nonzero_indices.tolist()
        values = nonzero_values.tolist()

        return SparseVector(indices=indices, values=values)

    elif response.status_code == 503:
        raise RuntimeError("A bge-m3-multi modell túlterhelt. Próbáld meg később újra.")
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")
      

"""
#dense vector lekerdezese
def get_chunk_id_from_embedding(query_embedding):
    #legközelebbi vektorok lekérdezése
    results = vector_db.query(vector=query_embedding, include_metadata=True,top_k=50,query_mode=QueryMode.DENSE)    
    chunk_ids = [result.metadata.get('chunk_id') for result in results]
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    chunk_ids = []
    for result in sorted_results:
        if result.score >= 0.86:
            #print(f"chunk_id: {result.metadata.get('chunk_id')}\nText: {result.metadata.get('text')}\nScore: {result.score}\n\n")
            chunk_ids.append(result.metadata.get('chunk_id'))

    print(len(chunk_ids)) 
    #print(chunk_ids)
    return chunk_ids"""
#hibrid lekerdezes: dense (sűrű) + sparse (ritka) vectorok és a függvny visszaadja a chunk_id-ket
def get_chunk_id_from_embedding(query_embedding, query_sparse_vector):
    # hibrid lekérdezés: dense + sparse
    results = vector_db.query(
        vector=query_embedding,
        sparse_vector=query_sparse_vector,
        fusion_algorithm=FusionAlgorithm.RRF,  # vagy FusionAlgorithm.DBSF
        include_metadata=True,
        top_k=50,
        query_mode=QueryMode.HYBRID
    )
    
    # chunk_id-k kiszűrése, duplikációk eltávolítása
    chunk_ids = list(dict.fromkeys(
        result.metadata.get('chunk_id') for result in results
    ))

    #print(chunk_ids)
    #print(len(chunk_ids))
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

    return grouped_by_chunk_id,len(chunk_ids)

def get_context_text(query_embedding,query_sparse_vector):
    try:
        #chunk_ids = get_chunk_id_from_embedding(query_embedding)
        chunk_ids = get_chunk_id_from_embedding(query_embedding,query_sparse_vector)
        if not chunk_ids:
            # nincs releváns chunk — fallback szöveg
            return (
                "Figyelem: Nem áll rendelkezésre releváns kontextus. "
                "Kérlek, válaszolj a saját tudásod alapján, ha tudsz, vagy jelezd, hogy nem tudsz válaszolni."
            ), 0

        grouped_results, top_k_size = query_by_chunk_id(chunk_ids)
        if not grouped_results:
            # nincs találat — fallback szöveg
            return (
                "Figyelem: Nem áll rendelkezésre releváns kontextus. "
                "Kérlek, válaszolj a saját tudásod alapján, ha tudsz, vagy jelezd, hogy nem tudsz válaszolni."
            ), 0

        # minden chunk_id-hoz tartozó szöveg összefűzése, chunk_order szerint
        context_parts = []
        for chunk_id in sorted(grouped_results.keys()):
            texts = [r.metadata.get('text','') for r in grouped_results[chunk_id]]
            text_joined = "\n".join(texts)
            context_parts.append(f"{chunk_id}\n{text_joined}")

        # chunk_id csoportokat elválasztjuk dupla sortöréssel
        context = "\n\n".join(context_parts)
        return context, top_k_size
    except Exception as e:
        print(f"Hiba a kontextus összeállítása során: {e}")
        # hiba esetén is fallback szöveg, a rendszer ne essen szét 
        return (
            "Figyelem: Nem áll rendelkezésre releváns kontextus (hiba történt a kontextus generálásakor). "
            "Kérlek, válaszolj a saját tudásod alapján, ha tudsz, vagy jelezd, hogy nem tudsz válaszolni."
        ), 0



#az LLM model segitségével választ generalok a feltett kérdésre
def get_llm_response(context, question):
    
    #client = genai.Client(api_key=api_key_pro)
    client = genai.Client(api_key=api_key)

    full_prompt = f"<context>{context}</context>Kérem, válaszoljon az alábbi kérdésre a fent megadott kontextus alapján, vedd ki a markdown formátumot:<user_query>{question}</user_query>\nVálasz:"
    try:
        response = client.models.generate_content(
            #model = "gemini-2.5-pro-preview-05-06",
            #model = "gemini-2.5-flash-preview-05-20",
            #model="gemini-2.0-flash-thinking-exp-01-21",
            model = "gemini-2.0-flash",
            #model = "gemini-2.5-flash-preview-04-17",
            #model = "gemini-1.5-flash",
            #model = "gemini-2.5-pro-preview-05-06",     
                 


            contents=[full_prompt]
        )
        return response.text
    except Exception as e:
        if "503" in str(e):
            raise RuntimeError(f"Az LLM modell túlterhelt. Próbáld meg később újra.")
        else:
            raise RuntimeError(f"LLM error: {str(e)}")
               
@app.route("/")
def init_load():
    global loading_started
    with loading_lock:
        if not loading_started:
            loading_started = True
            # háttérszálban van futattva, hogy ne blokkolja a válaszadást
            threading.Thread(target=load_all_vectors_to_list).start()
        if not loading_done:
                return render_template('loading.html')
    return redirect("/chatbot")

@app.route("/status")
def status():
    return jsonify({"done": loading_done})

@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        data = request.get_json()
        start_total = time.time()

        user_question = data.get("question", "").strip()

        if not user_question:
            return jsonify({"error": "Nincs megadva kérdés"}), 400
        try:
            start_embed = time.time()
            embedding = get_embedding(user_question) #OpenAI API-val átalakítja a szöveget embedding-gé (vektorrá)
            end_embed = time.time()
            t_embed = end_embed - start_embed

        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": "Unknown error: " + str(e)}), 500
       
        try:
            start_sparse_embed = time.time()
            query_sparse_vector = get_sparse_vector_from_query(user_question)
            end_sparse_embed = time.time()
            t_sparse_embed = end_sparse_embed - start_sparse_embed

        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": "Unknown error: " + str(e)}), 500
        
        try:
            start_ctx = time.time()
            # kontextus összeállítása a lekérdezett embedding alapján
            context, top_k_size = get_context_text(embedding,query_sparse_vector)
            end_ctx = time.time()
            t_ctx = end_ctx - start_ctx
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": "Unknown error: " + str(e)}), 500
        
        try:
            start_llm = time.time()
            resp = get_llm_response(context, user_question)
            end_llm = time.time()
            t_llm = end_llm - start_llm
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": "Unknown error: " + str(e)}), 500

        end_total = time.time()
        t_total = end_total - start_total


        print(f"\n--- Időmérések (másodpercben) ---")
        print(f"Embedding generálás:     {t_embed:.3f}s")
        print(f"Sparse Embedding generálás: {t_sparse_embed:3f}s")
        print(f"Kontextus összeállítás:  {t_ctx:.3f}s")
        print(f"LLM válasz: {t_llm:.3f}s")
        print(f"TELJES kérés feldolgozás: {t_total:.3f}s")
        print(f"top_k száma: {top_k_size}")
        
        

        return jsonify({"answer": resp})
        

    return render_template('index.html')

if __name__ == '__main__':
    #load_all_vectors_to_list()
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=10000 )
