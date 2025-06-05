from flask import Flask, jsonify, render_template, request, redirect
from upstash_vector import Index
import numpy as np
from collections import defaultdict
import os
import time
import openai
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


@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    print(f"Process ID: {os.getpid()}")
    if request.method == 'POST':
        

        data = request.get_json()
       
        user_question = data.get("question", "").strip()

        if not user_question:
            return jsonify({"error": "Nincs megadva kérdés"}), 400
        embedding = get_embedding(user_question) #OpenAI API-val átalakítja a szöveget embedding-gé (vektorrá)
        
        # kontextus összeállítása a lekérdezett embedding alapján
        context = get_context_text(embedding)
        
        resp = get_llm_response(context,user_question)
        

        
        
        return jsonify({"answer": resp})
    

    return render_template('index.html')

if __name__ == '__main__':
    #load_all_vectors_to_list()
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=10000)

