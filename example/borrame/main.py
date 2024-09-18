#!/usr/bin/python3

import pandas as pd

def find_unprocessed_data(all_files_csv_path, processed_csv_path):
    # Carregue os dados dos arquivos CSV
    df_all_files = pd.read_csv(all_files_csv_path)
    df_processed = pd.read_csv(processed_csv_path)

    # Encontre os dados que ainda não foram processados
    df_all_files['body'] = df_all_files['body'].astype(str)  # Certifique-se de que os caminhos são strings
    df_processed['body'] = df_processed['body'].astype(str)
    all_files_set = set(df_all_files['body'])
    processed_set = set(df_processed['body'])

    # Calcule a diferença entre conjuntos
    missing_data = all_files_set - processed_set

    # Filtre o DataFrame com base na diferença
    missing_df = df_all_files[df_all_files['body'].isin(missing_data)]

    # Ordene o DataFrame pela coluna 'body'
    sorted_missing_df = missing_df.sort_values(by='body').reset_index(drop=True)

    return sorted_missing_df

# Exemplo de uso
all_files_path = 'all_files.csv'
processed_path = 'processed_files.csv'
result_df = find_unprocessed_data(all_files_path, processed_path)
print(result_df)

