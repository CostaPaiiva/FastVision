import os # Importa o módulo os para interagir com o sistema operacional.
import sqlite3 # Importa o módulo sqlite3 para trabalhar com bancos de dados SQLite.
from typing import List, Tuple # Importa tipos para anotações de tipo.

DB_PATH = os.path.join("data", "app.db") # Define o caminho completo para o arquivo do banco de dados.

def init_db() -> None: # Define a função para inicializar o banco de dados.
    os.makedirs("data", exist_ok=True) # Cria o diretório 'data' se ele não existir.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados SQLite.
    cur = conn.cursor() # Cria um objeto cursor para executar comandos SQL.
    cur.execute(""" # Inicia a execução de um comando SQL multilinha.
        CREATE TABLE IF NOT EXISTS people ( # Cria a tabela 'people' se ela não existir.
            id INTEGER PRIMARY KEY AUTOINCREMENT, # Define 'id' como chave primária auto incrementada.
            name TEXT NOT NULL UNIQUE # Define 'name' como texto não nulo e único.
        )
    """) # Fecha o comando SQL.
    cur.execute(""" # Inicia a execução de outro comando SQL multilinha.
        CREATE TABLE IF NOT EXISTS images ( # Cria a tabela 'images' se ela não existir.
            id INTEGER PRIMARY KEY AUTOINCREMENT, # Define 'id' como chave primária auto incrementada.
            person_id INTEGER NOT NULL, # Define 'person_id' como inteiro não nulo.
            path TEXT NOT NULL, # Define 'path' como texto não nulo.
            FOREIGN KEY(person_id) REFERENCES people(id) # Define 'person_id' como chave estrangeira referenciando 'people(id)'.
        )
    """) # Fecha o comando SQL.
    conn.commit() # Salva as alterações no banco de dados.
    conn.close() # Fecha a conexão com o banco de dados.

def upsert_person(name: str) -> int: # Define a função para inserir ou obter uma pessoa pelo nome.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados.
    cur = conn.cursor() # Cria um objeto cursor.
    cur.execute("INSERT OR IGNORE INTO people(name) VALUES(?)", (name,)) # Insere o nome na tabela 'people' se não existir.
    conn.commit() # Salva a inserção.
    cur.execute("SELECT id FROM people WHERE name=?", (name,)) # Seleciona o id da pessoa pelo nome.
    pid = cur.fetchone()[0] # Obtém o id da pessoa.
    conn.close() # Fecha a conexão.
    return pid # Retorna o id da pessoa.

def add_image(person_id: int, path: str) -> None: # Define a função para adicionar uma imagem.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados.
    cur = conn.cursor() # Cria um objeto cursor.
    cur.execute("INSERT INTO images(person_id, path) VALUES(?,?)", (person_id, path)) # Insere a imagem com o id da pessoa e o caminho.
    conn.commit() # Salva a inserção.
    conn.close() # Fecha a conexão.

def list_people() -> List[Tuple[int, str]]: # Define a função para listar todas as pessoas.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados.
    cur = conn.cursor() # Cria um objeto cursor.
    cur.execute("SELECT id, name FROM people ORDER BY name") # Seleciona o id e o nome de todas as pessoas, ordenado por nome.
    rows = cur.fetchall() # Obtém todas as linhas resultantes.
    conn.close() # Fecha a conexão.
    return rows # Retorna a lista de pessoas.

def list_images() -> List[Tuple[int, int, str]]: # Define a função para listar todas as imagens.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados.
    cur = conn.cursor() # Cria um objeto cursor.
    cur.execute("SELECT id, person_id, path FROM images") # Seleciona o id, person_id e path de todas as imagens.
    rows = cur.fetchall() # Obtém todas as linhas resultantes.
    conn.close() # Fecha a conexão.
    return rows # Retorna a lista de imagens.

def delete_person(person_id: int) -> None: # Define a função para deletar uma pessoa e suas imagens.
    conn = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados.
    cur = conn.cursor() # Cria um objeto cursor.
    cur.execute("DELETE FROM images WHERE person_id=?", (person_id,)) # Deleta todas as imagens associadas à pessoa.
    cur.execute("DELETE FROM people WHERE id=?", (person_id,)) # Deleta a pessoa da tabela 'people'.
    conn.commit() # Salva as exclusões.
    conn.close() # Fecha a conexão.
