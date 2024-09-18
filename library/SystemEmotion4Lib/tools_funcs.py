#!/usr/bin/python3

import os
import re
import csv
from tqdm import tqdm

def natural_sort_key(s):
    """
    Gera uma chave de ordenação natural para strings.

    Esta função é usada para ordenar strings de forma "natural", ou seja, 
    considerando números no meio das strings de forma humana. Em vez de ordenar 
    lexicograficamente (onde 'file10' viria antes de 'file2'), a função divide a 
    string em partes numéricas e não numéricas para realizar a ordenação correta 
    ('file2' antes de 'file10').

    Parâmetros:
    -----------
    s : str
        A string que será processada para gerar a chave de ordenação natural.

    Retorna:
    --------
    list
        Uma lista de partes da string, onde as partes numéricas são convertidas 
        para inteiros e as partes textuais são convertidas para minúsculas, para 
        uso na ordenação natural.
    
    Exemplo de uso:
    ---------------
    >>> sorted(['file10', 'file2', 'file1'], key=natural_sort_key)
    ['file1', 'file2', 'file10']
    """
    # Função de chave para ordenação natural
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]




def list_png_in_match_subdir(directory,match_subdir='body',file_ext='.png'):
    """
    Lista todos os arquivos com uma extensão específica em subdiretórios correspondentes a um nome específico.

    Esta função percorre recursivamente o diretório fornecido e retorna uma lista de arquivos que correspondem à
    extensão fornecida (por padrão, arquivos .png) dentro de subdiretórios cujo nome contenha a string especificada 
    (por padrão, 'body').

    Parâmetros:
    -----------
    directory : str
        O caminho do diretório raiz onde a função começará a procurar os arquivos.
    match_subdir : str, opcional
        O nome parcial ou completo dos subdiretórios nos quais procurar os arquivos (o padrão é 'body').
    file_ext : str, opcional
        A extensão dos arquivos que devem ser listados (o padrão é '.png').

    Retorna:
    --------
    list
        Uma lista com os caminhos completos dos arquivos que correspondem à extensão e ao nome do subdiretório.
    
    Exemplo de uso:
    ---------------
    >>> list_png_in_match_subdir('/caminho/para/diretorio')
    ['/caminho/para/diretorio/body/file1.png', '/caminho/para/diretorio/body/file2.png']
    """
    png_files = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        #print('')
        #print('root:::',root)
        #print('dirs:::',dirs)
        #print('files:::',files)
        #print('')
        
        # Ordena os diretórios para garantir a ordem natural
        dirs.sort(key=natural_sort_key)
        # Verifica se o diretório atual é match_subdir
        if match_subdir in os.path.basename(root):
            # Coleta e ordena os arquivos .png com ordenação natural
            png_files.extend([os.path.join(root, file) for file in sorted(files, key=natural_sort_key) if file.endswith(file_ext)])
    
    # Retorna a lista final de arquivos .png ordenados naturalmente
    return sorted(png_files, key=natural_sort_key)



def file_exists_and_not_empty(filepath):
    """
    Verifica se um arquivo existe e tem peso diferente de zero.

    Args:
        filepath (str): O caminho do arquivo a ser verificado.

    Returns:
        bool: True se o arquivo existir e tiver tamanho maior que zero, False caso contrário.
    """
    # Verifica se o arquivo existe
    if os.path.isfile(filepath):
        # Verifica se o tamanho do arquivo é maior que zero
        return os.path.getsize(filepath) > 0
    else:
        return False


def verify_dataset_body_structure(png_files):
    """
    Verifica a estrutura de um dataset de arquivos PNG e seus arquivos associados.

    Esta função recebe uma lista de caminhos de arquivos PNG (body) e verifica se os arquivos correspondentes de 'face' 
    (com a mesma base de nome) e 'skeleton' (com a extensão .npy) estão presentes no diretório esperado. Os arquivos 'face' 
    e 'skeleton' são inferidos com base no local dos arquivos PNG. Se todos os arquivos estiverem presentes, a função 
    retorna `True` e uma lista de tuplas contendo os caminhos dos arquivos PNG, 'face', e 'skeleton'. Caso contrário, 
    retorna `False` e uma lista de tuplas com os arquivos ausentes.

    Parâmetros:
    -----------
    png_files : list of str
        Uma lista de caminhos para arquivos PNG a serem verificados.

    Retorna:
    --------
    tuple
        Um par de valores: 
        - Um valor booleano `True` se todos os arquivos PNG, 'face', e 'skeleton' existirem, ou `False` se algum deles faltar.
        - Uma lista de tuplas:
            - Se `True`, a lista contém tuplas no formato `(png_file, face_file, skeleton_file)` para cada arquivo válido.
            - Se `False`, a lista contém tuplas no mesmo formato, mas apenas para os conjuntos de arquivos que estão incompletos 
              (algum dos arquivos está faltando).

    Exemplo de uso:
    ---------------
    >>> png_files = ['dir1/body/file1.png', 'dir2/body/file2.png']
    >>> result, file_structure = verify_dataset_body_structure(png_files)
    >>> result
    True
    >>> file_structure
    [('dir1/body/file1.png', 'dir1/face/file1.png', 'dir1/skeleton/file1.npy'),
     ('dir2/body/file2.png', 'dir2/face/file2.png', 'dir2/skeleton/file2.npy')]

    """
    defective_files = []
    all_files_exist = True
    result_tuples = []
    
    for k in tqdm(range(len(png_files))):
        
        png_file=png_files[k];
        # Define paths for 'face' and 'skeleton' files
        face_file = os.path.join(os.path.dirname(png_file), '..', 'face', os.path.basename(png_file))
        skeleton_file = os.path.join(os.path.dirname(png_file), '..', 'skeleton', os.path.splitext(os.path.basename(png_file))[0] + '.npy')
        
        # Check if all files exist
        if file_exists_and_not_empty(png_file) and file_exists_and_not_empty(face_file) and file_exists_and_not_empty(skeleton_file):
            result_tuples.append((png_file, face_file, skeleton_file))
        else:
            all_files_exist = False
            defective_files.append((png_file, face_file, skeleton_file))
    
    # Return True if all files exist, otherwise False and list of defective tuples
    if all_files_exist:
        return True, result_tuples
    else:
        return False, defective_files

################################################################################


def save_dataset_list_in_csv(directory,file_list, csv_filename, my_func):
    """
    Salva uma lista de tuplas de arquivos em um arquivo CSV e processa os dados com uma função customizada.

    Esta função recebe uma lista de tuplas contendo três caminhos de arquivos (correspondentes a arquivos 'body', 'face' e 
    'skeleton'), processa os dados usando uma função fornecida pelo usuário (`my_func`), e grava os resultados em um arquivo CSV. 
    O arquivo CSV terá quatro colunas: os três caminhos de arquivo e o resultado processado.

    Parâmetros:
    -----------
    directory: str
        Directory path of reference
    
    file_list : list of tuple
        Uma lista de tuplas, onde cada tupla contém três strings que representam os caminhos dos arquivos 'body', 'face' e 'skeleton'.
    
    csv_filename : str
        O nome do arquivo CSV onde os dados serão salvos.
    
    my_func : function
        Uma função personalizada que recebe três parâmetros (os arquivos 'body', 'face' e 'skeleton') e retorna um valor, 
        que será salvo na quarta coluna do CSV.

    Retorna:
    --------
    None
        A função não retorna nenhum valor, mas salva o resultado em um arquivo CSV.

    Exemplo de uso:
    ---------------
    >>> def example_func(body_file, face_file, skeleton_file):
    ...     return f"Processed: {body_file}, {face_file}, {skeleton_file}"
    >>> file_list = [
    ...     ('body/file1.png', 'face/file1.png', 'skeleton/file1.npy'),
    ...     ('body/file2.png', 'face/file2.png', 'skeleton/file2.npy')
    ... ]
    >>> save_dataset_list_in_csv('/',file_list, 'output.csv', example_func)
    
    Isso gerará um arquivo 'output.csv' com o seguinte conteúdo:
    
    body,face,skeleton,label
    body/file1.png,face/file1.png,skeleton/file1.npy,Processed: body/file1.png, face/file1.png, skeleton/file1.npy
    body/file2.png,face/file2.png,skeleton/file2.npy,Processed: body/file2.png, face/file2.png, skeleton/file2.npy
    """
    # Abre o arquivo CSV para escrita
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Escreve o cabeçalho do CSV (opcional)
        csv_writer.writerow(['body', 'face', 'skeleton', 'label'])

        # Para cada tupla na file_list, processa e salva no CSV
        for file_tuple in file_list:
            png_file, face_file, skeleton_file = file_tuple
            # Processa os dados com my_func e obtém o resultado
            result = my_func(png_file, face_file, skeleton_file)
            # Escreve a linha no CSV
            csv_writer.writerow([   os.path.relpath(png_file,directory), 
                                    os.path.relpath(face_file,directory), 
                                    os.path.relpath(skeleton_file,directory), 
                                    result])





def save_dataset_list_in_csv_batch(directory,file_list, csv_filename, my_batch_func, batch_size=64):
    """
    Salva uma lista de tuplas de arquivos em um arquivo CSV e processa os dados em lotes usando uma função customizada.

    Esta função recebe uma lista de tuplas contendo três caminhos de arquivos (correspondentes a arquivos 'body', 'face' e 
    'skeleton'), processa os dados em lotes usando uma função fornecida pelo usuário (`my_batch_func`), e grava os resultados em um 
    arquivo CSV. O arquivo CSV terá quatro colunas: os três caminhos de arquivo e o resultado processado.

    Parâmetros:
    -----------
    directory: str
        Directory path of reference
    
    file_list : list of tuple
        Uma lista de tuplas, onde cada tupla contém três strings que representam os caminhos absolutos dos arquivos 'body', 'face' e 'skeleton'.
    
    csv_filename : str
        O nome do arquivo CSV onde os path relativos serão salvos.
    
    my_batch_func : function
        Uma função personalizada que recebe uma lista de tuplas (cada tupla contém 'body', 'face', 'skeleton') e retorna uma 
        lista de resultados, que será salvo na quarta coluna do CSV.
    
    batch_size : int, opcional
        O número de elementos que serão processados em cada lote (batch). O valor padrão é 64.

    Retorna:
    --------
    None
        A função não retorna nenhum valor, mas salva o resultado em um arquivo CSV.

    Exemplo de uso:
    ---------------
    >>> def example_func(batch):
    ...     return [f"Processed: {b}, {f}, {s}" for b, f, s in batch]
    
    >>> file_list = [
    ...     ('/absolute/body/file1.png', '/absolute/face/file1.png', '/absolute/skeleton/file1.npy'),
    ...     ('/absolute/body/file2.png', '/absolute/face/file2.png', '/absolute/skeleton/file2.npy')
    ... ]
    >>> save_dataset_list_in_csv_batch('/absolute',file_list, 'output.csv', example_func, batch_size=2)
    
    Isso gerará um arquivo 'output.csv' com o seguinte conteúdo:
    
    body,face,skeleton,label
    body/file1.png,face/file1.png,skeleton/file1.npy,Processed: body/file1.png, face/file1.png, skeleton/file1.npy
    body/file2.png,face/file2.png,skeleton/file2.npy,Processed: body/file2.png, face/file2.png, skeleton/file2.npy
    """
    # Escreve o cabeçalho apenas se o arquivo não existe
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['body', 'face', 'skeleton', 'label'])

    # Processa os dados em lotes (batch)
    for i in tqdm(range(0, len(file_list), batch_size)):
        batch = file_list[i:i+batch_size]
        
        # Processa o lote com my_batch_func
        results = my_batch_func(batch)
        
        # Abre o arquivo CSV em modo append ('a') para adicionar os novos resultados
        with open(csv_filename, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for file_tuple, result in zip(batch, results):
                csv_writer.writerow([
                    os.path.relpath(file_tuple[0], directory),
                    os.path.relpath(file_tuple[1], directory),
                    os.path.relpath(file_tuple[2], directory),
                    result
                ])


