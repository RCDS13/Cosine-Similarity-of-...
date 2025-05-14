# -*- coding: utf-8 -*-
"""
Sistema de Recomendação de Jogos Steam com Similaridade de Cosseno
Versão Melhorada com Tratamento de Erros, Cache e Performance
"""

# ==================== INSTALAÇÃO DE DEPENDÊNCIAS ====================
import os
import sys
import subprocess
import hashlib
from time import sleep
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from io import StringIO


def install_dependencies():
    """Instala dependências com tratamento de erros"""
    dependencies = [
        "pandas", "scikit-learn", "nltk",
        "requests", "joblib", "numpy", "matplotlib"
    ]

    print('=' * 80)
    print('Verificando dependências...')

    for package in dependencies:
        try:
            __import__(package)
            print(f"{package:20} ✓ Já instalado")
        except ImportError:
            print(f"{package:20} ✗ Instalando...", end=' ')
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("✓ Concluído!")
            except subprocess.CalledProcessError:
                print("✗ Falha na instalação")

    print('=' * 80)
    print('Verificação de dependências concluída!')
    print('=' * 80)


# Executa a instalação de dependências
install_dependencies()

# ==================== IMPORTAÇÕES ====================
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from joblib import dump, load
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== CONFIGURAÇÕES ====================
# URLs dos datasets
STEAM_DATA_URL = 'https://raw.githubusercontent.com/RCDS13/Steam-Game-Similarity-Calculation-Agorithm/main/steam.csv'
STEAM_MODIFIED_URL = 'https://raw.githubusercontent.com/RCDS13/Steam-Game-Similarity-Calculation-Agorithm/main/steam_modified.csv'

# Configurações de diretórios
CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurações do pandas
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


# ==================== FUNÇÕES DE CARREGAMENTO DE DADOS ====================
def download_with_retry(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Baixa conteúdo com retry automático"""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": "SteamGameRecommender/1.0",
        "Accept": "text/csv"
    }

    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar {url}: {str(e)}")
        return None


def load_and_preprocess_data() -> Optional[pd.DataFrame]:
    """Carrega e prepara os dados com tratamento de erros e cache"""
    cache_file = CACHE_DIR / "steam_data_cache.csv"

    # Tenta carregar do cache local primeiro
    if cache_file.exists():
        print("Carregando dados do cache local...")
        try:
            return pd.read_csv(cache_file)
        except Exception as e:
            print(f"Erro ao ler cache: {str(e)}")

    # Baixa e processa os dados
    print("Baixando dados do GitHub...")
    response = download_with_retry(STEAM_DATA_URL)
    if response is None:
        return None

    try:
        # Carrega e processa os dados
        df = pd.read_csv(StringIO(response.text))

        # Remove colunas não necessárias
        cols_to_drop = [
            'release_date', 'english', 'platforms', 'required_age',
            'achievements', 'positive_ratings', 'negative_ratings',
            'average_playtime', 'median_playtime', 'owners', 'price'
        ]
        df["name"] = df["name"].str.replace(r'®|™', '', regex=True)
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Salva em cache (formato CSV)
        df.to_csv(cache_file, index=False)
        print("Dados salvos em cache local.")

        return df

    except Exception as e:
        print(f"Erro ao processar dados: {str(e)}")
        return None


# ==================== PROCESSAMENTO TF-IDF ====================
def create_tfidf_matrix(df: pd.DataFrame) -> Tuple[Any, Any, Any, pd.DataFrame]:
    """Cria matriz TF-IDF com cache robusto"""
    if df is None:
        return None, None, None, None

    # Verifica e cria colunas necessárias
    required_columns = ['categories', 'genres', 'steamspy_tags', 'name', 'publisher', 'developer']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''  # Cria a coluna com valores vazios se não existir

    # Cria a coluna combined_text
    df['combined_text'] = (
            df['categories'].fillna('') + ' ' +
            df['genres'].fillna('') + ' ' +
            df['steamspy_tags'].fillna('') + ' ' +
            df['name'].fillna('') + ' ' +
            df['publisher'].fillna('') + ' ' +
            df['developer'].fillna('')
    ).str.replace(r'\s+', ' ', regex=True)

    # Gera hash único para os parâmetros atuais
    params_hash = hashlib.md5((
                                      str(df['combined_text'].values.tobytes()) +
                                      "800" + "1,2"  # max_features + ngram_range
                              ).encode()).hexdigest()

    cache_file = CACHE_DIR / f"tfidf_cache_{params_hash}.joblib"

    # Tenta carregar do cache
    if cache_file.exists():
        try:
            print("Carregando matriz TF-IDF do cache...")
            cached = load(cache_file)
            return cached['tfidf_matrix'], cached['vectorizer'], cached['cosine_sim'], df
        except Exception as e:
            print(f"Cache corrompido ({str(e)}). Recriando...")

    # Processamento dos dados
    print("Processando dados (pode demorar alguns minutos)...")

    # Configura stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    custom_stopwords = {
        'game', 'play', 'player', 'players', 'multi', 'single',
        'achievements', 'controller', 'partial', 'steam', 'support',
        'early', 'access', 'windows', 'mac', 'linux'
    }
    stop_words.update(custom_stopwords)

    # Configura o vetorizador TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=800,
        token_pattern=r'(?u)\b[a-zA-Z0-9_]{2,}\b',
        ngram_range=(1, 2),
        dtype=np.float32
    )

    # Pré-processamento do texto (agora corretamente indentado)
    df['processed_text'] = (
        df['combined_text']
        .str.lower()
        .str.replace('[;,.|]', ' ', regex=True)
        .apply(lambda x: ' '.join(sorted(set(x.split()), key=x.split().index)))
    )

    # Cria a matriz TF-IDF
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Salva no cache
    try:
        dump({
            'tfidf_matrix': tfidf_matrix,
            'vectorizer': vectorizer,
            'cosine_sim': cosine_sim
        }, cache_file)
        print(f"Matriz TF-IDF salva em cache: {cache_file}")
    except Exception as e:
        print(f"Erro ao salvar cache: {str(e)}")

    return tfidf_matrix, vectorizer, cosine_sim, df


# ==================== FUNÇÕES DE RECOMENDAÇÃO ====================
def find_similar_games(game_name: str, cosine_sim: np.ndarray, df: pd.DataFrame, top_n: int = 10) -> Optional[
    pd.DataFrame]:
    """Busca jogos similares com tratamento robusto"""
    if cosine_sim is None or df is None:
        return None

    try:
        # Busca case insensitive e tolerante a erros
        matches = df[df['name'].str.lower().str.contains(game_name.lower(), na=False)]

        if len(matches) == 0:
            print(f"\nJogo '{game_name}' não encontrado.")
            print("Sugestão: Verifique a ortografia ou veja alguns jogos populares:")
            print(df.sample(5)[['name']].to_string(index=False, header=False))
            return None

        # Seleção do jogo
        if len(matches) > 1:
            print("\nVários jogos encontrados. Escolha um:")
            for idx, row in matches.reset_index(drop=True).iterrows():
                print(f"{idx + 1}: {row['name']}")

            try:
                choice = int(input("\nDigite o número do jogo desejado: ")) - 1
                selected_index = matches.index[choice if 0 <= choice < len(matches) else 0]
            except:
                print("Entrada inválida. Usando o primeiro jogo da lista.")
                selected_index = matches.index[0]
        else:
            selected_index = matches.index[0]

        # Obtém recomendações
        game_name = df.loc[selected_index, 'name']
        print(f"\nBuscando jogos similares a: {game_name}")

        sim_scores = list(enumerate(cosine_sim[selected_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        # Formata resultados
        result = df.iloc[[i[0] for i in sim_scores]][['name', 'genres']].copy()
        result['similarity'] = [round(score, 3) for _, score in sim_scores]
        result['genres'] = result['genres'].str.replace(';', ', ')

        return result.reset_index(drop=True)

    except Exception as e:
        print(f"Erro ao buscar jogos similares: {str(e)}")
        return None


# ==================== INTERFACE GRÁFICA ====================
class SteamRecommenderGUI:
    def __init__(self, root: tk.Tk, df: pd.DataFrame, cosine_sim: np.ndarray):
        self.root = root
        self.df = df
        self.cosine_sim = cosine_sim
        self.selected_game_index: Optional[int] = None

        try:
            self.setup_ui()
            self.setup_styles()
            self.update_random_games()
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao inicializar interface: {str(e)}")
            self.root.destroy()
            raise

    def setup_ui(self) -> None:
        """Configura a interface gráfica"""
        self.root.title("Recomendação de Jogos Steam")
        self.root.geometry("1100x700")

        # Container principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Área de busca
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Buscar jogo:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame, width=40)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Buscar", command=self.search_games).pack(side=tk.LEFT)

        # Exibição de jogos aleatórios
        random_frame = ttk.LabelFrame(main_frame, text="Jogos Aleatórios")
        random_frame.pack(fill=tk.X, pady=5)
        self.random_games_label = ttk.Label(random_frame, text="", wraplength=800)
        self.random_games_label.pack()

        # Área de resultados
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview para resultados
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=('name', 'genres', 'similarity'),
            show='headings'
        )
        self.results_tree.heading('name', text='Nome do Jogo')
        self.results_tree.heading('genres', text='Gêneros')
        self.results_tree.heading('similarity', text='Similaridade')

        self.results_tree.column('name', width=300)
        self.results_tree.column('genres', width=400)
        self.results_tree.column('similarity', width=100, anchor='center')

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Área do gráfico
        graph_frame = ttk.LabelFrame(main_frame, text="Gráfico de Similaridade")
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Eventos
        self.results_tree.bind('<<TreeviewSelect>>', self.on_game_select)
        self.search_entry.bind('<Return>', lambda event: self.search_games())

    def setup_styles(self) -> None:
        """Configura os estilos visuais"""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        style.configure('Treeview', rowheight=25, font=('Arial', 9))
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))

    def update_random_games(self) -> None:
        """Atualiza a exibição de jogos aleatórios"""
        sample = self.df.sample(5)['name'].tolist()
        self.random_games_label.config(text=", ".join(sample))

    def search_games(self) -> None:
        """Realiza a busca por jogos"""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Aviso", "Por favor, digite o nome de um jogo")
            return

        # Limpa resultados anteriores
        self.results_tree.delete(*self.results_tree.get_children())

        try:
            # Busca os jogos
            matches = self.df[self.df['name'].str.lower().str.contains(query.lower(), na=False)]

            if len(matches) == 0:
                messagebox.showinfo("Não encontrado", f"Nenhum jogo encontrado para '{query}'")
                return

            if len(matches) > 1:
                self.show_selection_dialog(matches)
            else:
                self.show_similar_games(matches.index[0])
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na busca: {str(e)}")

    def show_selection_dialog(self, matches: pd.DataFrame) -> None:
        """Mostra diálogo de seleção para múltiplos resultados"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Selecione um Jogo")
        dialog.geometry("400x300")

        ttk.Label(dialog, text="Múltiplos jogos encontrados. Selecione um:").pack(pady=10)

        listbox = tk.Listbox(dialog, font=('Arial', 10))
        for idx, row in matches.iterrows():
            listbox.insert(tk.END, row['name'])
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def on_select():
            if listbox.curselection():
                selected_index = matches.index[listbox.curselection()[0]]
                dialog.destroy()
                self.show_similar_games(selected_index)

        ttk.Button(dialog, text="Selecionar", command=on_select).pack(pady=10)
        dialog.grab_set()

    def show_similar_games(self, game_index: int) -> None:
        """Mostra jogos similares ao selecionado"""
        self.selected_game_index = game_index
        game_name = self.df.loc[game_index, 'name']

        try:
            # Obtém jogos similares
            sim_scores = list(enumerate(self.cosine_sim[game_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

            # Limpa e preenche a treeview
            self.results_tree.delete(*self.results_tree.get_children())
            for idx, score in sim_scores:
                game_data = self.df.iloc[idx]
                self.results_tree.insert('', 'end',
                                         values=(
                                             game_data['name'],
                                             game_data['genres'].replace(';', ', '),
                                             f"{score:.3f}"
                                         )
                                         )

            # Atualiza o gráfico
            self.update_similarity_graph(game_index)

            # Seleciona o primeiro item
            if self.results_tree.get_children():
                self.results_tree.selection_set(self.results_tree.get_children()[0])

        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao buscar similares: {str(e)}")

    def update_similarity_graph(self, game_index: int) -> None:
        """Atualiza o gráfico de similaridade"""
        self.ax.clear()

        game_name = self.df.loc[game_index, 'name']
        sim_scores = list(enumerate(self.cosine_sim[game_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

        games = [self.df.iloc[i[0]]['name'] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        y_pos = np.arange(len(games))
        bars = self.ax.barh(y_pos, scores, color='skyblue')

        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(games)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Nível de Similaridade')
        self.ax.set_title(f'Jogos similares a "{game_name}"')

        for bar in bars:
            width = bar.get_width()
            self.ax.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2f}',
                         ha='left', va='center')

        self.canvas.draw()

    def on_game_select(self, event: tk.Event) -> None:
        """Trata a seleção de um jogo na lista"""
        selected = self.results_tree.selection()
        if selected:
            # Implemente ações adicionais aqui se necessário
            pass


# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    """Função principal do aplicativo"""
    try:
        print("\n" + "=" * 80)
        print("Iniciando Sistema de Recomendação de Jogos Steam")
        print("=" * 80 + "\n")

        # Carrega e processa os dados
        print("Carregando dados Steam...")
        df = load_and_preprocess_data()

        if df is None:
            messagebox.showerror("Erro", "Não foi possível carregar os dados")
            return

        # Processamento TF-IDF
        print("Processando dados para recomendação...")
        tfidf_matrix, vectorizer, cosine_sim, df = create_tfidf_matrix(df)

        # Configuração da janela principal
        root = tk.Tk()

        # Centraliza a janela
        window_width = 1100
        window_height = 700
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        # Inicia a interface
        SteamRecommenderGUI(root, df, cosine_sim)
        root.mainloop()

    except Exception as e:
        print(f"\nErro fatal: {str(e)}")
        messagebox.showerror("Erro Fatal", f"O aplicativo encontrou um erro:\n{str(e)}")
    finally:
        print("\n" + "=" * 80)
        print("Sistema encerrado")
        print("=" * 80)


if __name__ == "__main__":
    main()