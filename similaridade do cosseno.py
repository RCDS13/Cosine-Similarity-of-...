# 00 - INSTALANDO DEPENDÊNCIAS

from time import sleep
print('='*80)

print('Instalando dependências...')
sleep(2)

print('='*80)

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "nltk", "requests", "joblib", "numpy", "matplotlib"])

print('='*80)

print('Dependências instaladas! Importando...')

print('='*80)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import requests
from io import StringIO
from joblib import dump, load
import hashlib
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# PARTE 01 - EXCLUSÃO DE COLUNAS IRRELEVANTES DO DATASET ORIGINAL (SEM TEXTOS ÚTEIS)

# Carregar o dataset diretamente do GitHub
url = 'https://raw.githubusercontent.com/RCDS13/Steam-Game-Similarity-Calculation-Agorithm/refs/heads/main/steam.csv'
urlmod = 'https://raw.githubusercontent.com/RCDS13/Steam-Game-Similarity-Calculation-Agorithm/refs/heads/main/steam_modified.csv'

df = pd.read_csv(url)

# Lista das colunas a eliminar
colunas_para_eliminar = [
    'release_date', 'english', 'platforms', 'required_age',
    'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime',
    'median_playtime', 'owners', 'price'
]

df["name"] = df["name"].str.replace(r'®|™', '', regex=True)

# Eliminar as colunas indesejadas
df = df.drop(columns=colunas_para_eliminar)

# Exibir as primeiras linhas do Dataset modificado

print("\nExemplo do começo do Dataset modificado:")
print('='*80)
print(df.head())
print('='*80)

# Configurações
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
nltk.download('stopwords')
CACHE_DIR = "tfidf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gerando hash para otimização do algoritmo
def get_data_hash(df):
    return hashlib.md5(df['combined_text'].values.tobytes()).hexdigest()

# Carrega e prepara os dados com otimização de memória
def load_and_preprocess_data():
    cache_file = 'steam_modified_cached.csv'

    # Se já existe local, usa ele
    if os.path.exists(cache_file):
        print("Carregando dataset do cache local...")
        usecols = ['appid', 'name', 'categories', 'genres', 'steamspy_tags', 'developer', 'publisher']
        df = pd.read_csv(cache_file, usecols=usecols, low_memory=False)
    else:
        try:
            print("Baixando dataset do GitHub...")
            response = requests.get(urlmod, timeout=10)
            response.raise_for_status()

            usecols = ['appid', 'name', 'categories', 'genres', 'steamspy_tags', 'developer', 'publisher']
            df = pd.read_csv(StringIO(response.text), usecols=usecols, low_memory=False)
            df.to_csv(cache_file, index=False)
            print("Download concluído. Dataset salvo em cache.")
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return None

    # Criação do campo de texto combinado
    df['combined_text'] = (
        df['categories'].fillna('') + ' ' +
        df['genres'].fillna('') + ' ' +
        df['steamspy_tags'].fillna('') + ' ' +
        df['name'].fillna('') + ' ' +
        df['publisher'].fillna('') + ' ' +
        df['developer'].fillna('')
    ).str.replace(r'\s+', ' ', regex=True)

    return df

# Cria matriz TF-IDF e salva dataset com features
def create_tfidf_matrix(df):
    if df is None:
        return None, None, None, None

    data_hash = get_data_hash(df)
    cache_file = os.path.join(CACHE_DIR, f"{data_hash}.joblib")
    output_file = os.path.join(OUTPUT_DIR, f"steam_with_features_{data_hash[:8]}.csv")

    # Tenta carregar do cache
    if os.path.exists(cache_file):
        print("Carregando dados do cache...")
        cached = load(cache_file)

        # Verifica se o CSV de output já existe
        if not os.path.exists(output_file):
            save_dataset_with_features(cached['vectorizer'], cached['tfidf_matrix'], df, output_file)

        return cached['tfidf_matrix'], cached['vectorizer'], cached['cosine_sim'], df

    print("Processando dados (pode demorar alguns minutos)...")

    # PARTE 03 - APLICA A TÉCNICA DE STOP WORDS AO DATASET
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {
        'game', 'play', 'player', 'players', 'multi', 'single',
        'achievements','controller', 'partial', 'steam', 'support',
        'early', 'access', 'windows', 'mac', 'linux'
    }
    stop_words.update(custom_stopwords)

    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=800,
        token_pattern=r'(?u)\b[a-zA-Z0-9_]{2,}\b',
        ngram_range=(1, 2),
        dtype=np.float32
    )

    # Pré-processamento
    df['processed_text'] = df['combined_text'].str.lower().str.replace('[;,.|]', ' ', regex=True)
    df['processed_text'] = df['processed_text'].apply(
        lambda x: ' '.join(sorted(set(x.split()), key=x.split().index)))

    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    # PARTE 04 - CÁLCULO DE SIMILARIDADE E APRESENTAÇÃO DOS RESULTADOS NO CONSOLE

    # Pré-calcula a matriz de similaridade
    print('''    Calculando similaridades... 
    exibindo insights do TF-IDF... 
    aguarde a interface inicializar...''')
    print('=' * 80)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # PARTE 05 - INSIGHTS DO TF-IDF

    # Verifica a qualidade do processamento textual
    def debug_similarity():
        print("\n=== DIAGNÓSTICO DO MODELO ===")

        # Amostra de textos processados
        print("\nAmostra de textos processados:")
        sample = df.sample(3)
        for idx, row in sample.iterrows():
            print(f"\nJogo: {row['name']}")
            print(f"Texto original: {row['combined_text']}")
            print(f"Processado: {row['processed_text']}")

        # Verifica termos mais importantes
        print("\nTop 10 termos no TF-IDF:")
        feature_names = vectorizer.get_feature_names_out()
        print(feature_names[:10])

        # Testa similaridade entre jogos conhecidos
        test_pairs = [
            ("Portal", "Portal 2"),
            ("Call of Duty 2", "Call of Duty"),
            ("Left 4 Dead", "Left 4 Dead 2"),
        ]

        for game1, game2 in test_pairs:
            try:
                idx1 = df[df['name'].str.contains(game1, case=False)].index[0]
                idx2 = df[df['name'].str.contains(game2, case=False)].index[0]
                sim = cosine_sim[idx1, idx2]
                print(f"\nSimilaridade '{df.loc[idx1, 'name']}' vs '{df.loc[idx2, 'name']}': {sim:.2f}")

                # Mostra termos em comum
                terms1 = set(row['processed_text'].split())
                terms2 = set(df.loc[idx2, 'processed_text'].split())
                common = terms1 & terms2
                print(f"Termos em comum: {common if common else 'Nenhum'}")

            except Exception as e:
                print(f"\nErro ao comparar {game1} e {game2}: {str(e)}")

    # Executa os insights
    debug_similarity()

    # Salva no cache
    dump({
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'cosine_sim': cosine_sim
    }, cache_file)

    # Salva dataset com features
    save_dataset_with_features(vectorizer, tfidf_matrix, df, output_file)

    return tfidf_matrix, vectorizer, cosine_sim, df


def save_dataset_with_features(vectorizer, tfidf_matrix, df, output_file):
    """Salva o dataset original com as features TF-IDF como colunas"""


    # Converte a matriz esparsa para Dataset
    features_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix,
        columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()])

    # Combina com os dados originais
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # Salvamento usado apenas para visualização do Dataset. (está no .rar do repositório)
    #result_df.to_csv(output_file, index=False)
    #print(f"Dataset com features salvo em: {output_file}")


# Busca por jogos similares com seleção numérica
def find_similar_games(game_name, cosine_sim, df, top_n=10):
    if cosine_sim is None or df is None:
        return None

    # Alternativas para 'jogo não encontrado'
    matches = df[df['name'].str.lower().str.contains(game_name.lower())]

    if len(matches) == 0:
        print(f"\nJogo '{game_name}' não encontrado.")
        print("Sugestão: Verifique a ortografia ou veja alguns jogos populares:")
        print(df.sample(5)[['name']].to_string(index=False, header=False))
        return None

    if len(matches) > 1:
        print("\nVários jogos encontrados. Escolha um:")
        matches = matches.reset_index(drop=True)
        for idx, row in matches.iterrows():
            print(f"{idx + 1}: {row['name']}")

        try:
            choice = int(input("\nDigite o número do jogo desejado: ")) - 1
            if 0 <= choice < len(matches):
                selected_index = matches.index[choice]
            else:
                print("Número inválido. Usando o primeiro jogo da lista.")
                selected_index = matches.index[0]
        except:
            print("Entrada inválida. Usando o primeiro jogo da lista.")
            selected_index = matches.index[0]
    else:
        selected_index = matches.index[0]

    game_name = df.loc[selected_index, 'name']
    print(f"\nBuscando jogos similares a: {game_name}")

    # Usa a matriz de similaridade pré-calculada
    sim_scores = list(enumerate(cosine_sim[selected_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    game_indices = [i[0] for i in sim_scores]

    # Formata resultados
    result = df.iloc[game_indices][['name', 'genres']].copy()
    result['similarity'] = [round(score, 3) for _, score in sim_scores]
    result['genres'] = result['genres'].str.replace(';', ', ')

    return result.reset_index(drop=True)


def main():
    print("Carregando dados Steam...")
    df = load_and_preprocess_data()

    if df is None:
        return

    # Carrega matriz TF-IDF e similaridades
    tfidf_matrix, vectorizer, cosine_sim, df = create_tfidf_matrix(df)

    print("\nSistema de Recomendação de Jogos Steam")
    print("-"*35)

    while True:
        print("\nJogos disponíveis (amostra aleatória):")
        print(df.sample(5)[['name']].to_string(index=False, header=False))

        game_name = input("\nDigite parte do nome do jogo (ou 'sair' para terminar): ").strip()

        if game_name.lower() == 'sair':
            break

        similar_games = find_similar_games(game_name, cosine_sim, df)

        if similar_games is not None:
            print(f"\nTop 10 jogos similares:")
            print(similar_games.to_string(index=False))

# PARTE 05 - CRIAÇÃO DA INTERFACE
class SteamRecommenderGUI:
    def __init__(self, root, df, cosine_sim):
        self.root = root
        self.df = df
        self.cosine_sim = cosine_sim
        self.selected_game_index = None

        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        self.root.title("Recomendação de jogos da Steam")
        self.root.geometry("1100x700")

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Busque o jogo:").pack(side=tk.LEFT)

        self.search_entry = ttk.Entry(search_frame, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)

        search_btn = ttk.Button(search_frame, text="Busca", command=self.search_games)
        search_btn.pack(side=tk.LEFT)

        # Random games display
        random_frame = ttk.LabelFrame(main_frame, text="Exemplo de jogos (aleatório)")
        random_frame.pack(fill=tk.X, pady=5)

        self.random_games_label = ttk.Label(random_frame, text="", wraplength=800)
        self.random_games_label.pack()
        self.update_random_games()

        # Results display
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for results
        self.results_tree = ttk.Treeview(results_frame, columns=('name', 'genres', 'similarity'), show='headings')
        self.results_tree.heading('name', text='Game Name')
        self.results_tree.heading('genres', text='Genres')
        self.results_tree.heading('similarity', text='Similarity')

        self.results_tree.column('name', width=300)
        self.results_tree.column('genres', width=400)
        self.results_tree.column('similarity', width=100, anchor='center')

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Graph frame
        graph_frame = ttk.LabelFrame(main_frame, text="Gráfico de similaridade")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self.on_game_select)

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Treeview', rowheight=25)

    def update_random_games(self):
        sample_games = self.df.sample(5)['name'].tolist()
        self.random_games_label.config(text=", ".join(sample_games))

    def search_games(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Atenção", "Insira o nome de um jogo")
            return

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Find matching games
        matches = self.df[self.df['name'].str.lower().str.contains(query.lower())]

        if len(matches) == 0:
            messagebox.showinfo("Not Found", f"No games found matching '{query}'")
            return

        if len(matches) > 1:
            self.show_selection_dialog(matches)
        else:
            self.show_similar_games(matches.index[0])

    def show_selection_dialog(self, matches):
        dialog = tk.Toplevel(self.root)
        dialog.title("Selecione o jogo")
        dialog.geometry("400x300")

        label = ttk.Label(dialog, text="Diversos jogos encontrados. Selecione um:")
        label.pack(pady=10)

        listbox = tk.Listbox(dialog)
        for idx, row in matches.iterrows():
            listbox.insert(tk.END, row['name'])
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def on_select():
            selection = listbox.curselection()
            if selection:
                selected_index = matches.index[selection[0]]
                dialog.destroy()
                self.show_similar_games(selected_index)

        select_btn = ttk.Button(dialog, text="Select", command=on_select)
        select_btn.pack(pady=10)

        dialog.grab_set()
        dialog.wait_window()

    def show_similar_games(self, game_index):
        self.selected_game_index = game_index
        game_name = self.df.loc[game_index, 'name']

        # Get similar games
        sim_scores = list(enumerate(self.cosine_sim[game_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add results to treeview
        for idx, score in sim_scores:  # Aqui score já é o valor de similaridade (não uma lista)
            game_data = self.df.iloc[idx]
            self.results_tree.insert('', 'end',
                                     values=(game_data['name'],
                                             game_data['genres'].replace(';', ', '),
                                             f"{score:.3f}"))  # Usamos score diretamente

        # Update graph
        self.update_similarity_graph(game_index)

        # Select first item
        if len(self.results_tree.get_children()) > 0:
            self.results_tree.selection_set(self.results_tree.get_children()[0])

    def update_similarity_graph(self, game_index):
        plt.rcParams['font.sans-serif'] = [
            'SimHei',  # Windows
            'Microsoft YaHei',  # Windows alternativo
            'WenQuanYi Zen Hei',  # Linux
            'AppleGothic',  # Mac
            'Noto Sans CJK JP'  # Fonte universal
        ]
        plt.rcParams['axes.unicode_minus'] = False  # Corrige exibição de sinais negativos

        self.ax.clear()

        game_name = self.df.loc[game_index, 'name']
        sim_scores = list(enumerate(self.cosine_sim[game_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10

        games = [self.df.iloc[i[0]]['name'] for i in sim_scores]
        scores = [i[1] for i in sim_scores]  # i[1] contém o valor de similaridade

        y_pos = np.arange(len(games))
        bars = self.ax.barh(y_pos, scores, color='skyblue')

        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(games)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Nível de similaridade')
        self.ax.set_title(f'Jogos similares à "{game_name}"')

        for bar in bars:
            width = bar.get_width()
            self.ax.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2f}',
                         ha='left', va='center')

        self.canvas.draw()

    def on_game_select(self, event):
        selected_item = self.results_tree.selection()
        if selected_item:
            pass

def main_gui():
    print("Carregando dados Steam...")
    df = load_and_preprocess_data()

    if df is None:
        return

    # Carrega a matrix IF-IDF e similaridades
    tfidf_matrix, vectorizer, cosine_sim, df = create_tfidf_matrix(df)

    # Cria GUI
    root = tk.Tk()
    SteamRecommenderGUI(root, df, cosine_sim)
    root.mainloop()

if __name__ == "__main__":
    main_gui()
