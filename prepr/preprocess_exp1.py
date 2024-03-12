import pandas as pd
import os
import numpy as np

# Ottiene un elenco di tutti i file CSV nella directory specificata
cartella = r'C:\Users\Utente\Desktop\TIROCINIO\python\project\directory_drivers'
elenco_file = [f for f in os.listdir(cartella) if f.endswith('.csv')]

def carica_csv_e_aggiungi_driver(file, cartella):
    """Carica un file CSV in un DataFrame e aggiunge una colonna per identificare il driver tester."""
    df = pd.read_csv(os.path.join(cartella, file))
    df['driver'] = os.path.splitext(file)[0]
    return df


'''def create_phase(df):
    """Crea una nuova colonna 'Fase' nel DataFrame."""
    # Inizializza il contatore delle fasi e la Drive corrente
    phase_counter = 1
    current_drive = df.loc[0, 'Drive']

    # Itera su tutte le righe del dataframe
    for i in range(len(df)):
        # Se la Drive cambia, resetta il contatore delle fasi
        if df.loc[i, 'Drive'] != current_drive:
            phase_counter = 1
            current_drive = df.loc[i, 'Drive']
        # Altrimenti, se lo stimolo cambia e la Drive è tra 5 e 7, incrementa il contatore delle fasi
        elif i > 0 and df.loc[i, 'Stimulus'] != df.loc[i - 1, 'Stimulus'] and df.loc[i, 'Drive'] in [5, 6, 7]:
            phase_counter += 1
        # Se la Drive è tra 0, 1, 2 o 3, assegna un valore personalizzato alla colonna 'Fase'
        if df.loc[i, 'Drive'] == 1:
            df.loc[i, 'Fase'] = 'baseline'
        elif df.loc[i, 'Drive'] == 2:
            df.loc[i, 'Fase'] = 'guida pratica'
        elif df.loc[i, 'Drive'] == 3:
            df.loc[i, 'Fase'] = 'relaxing drive'
        elif df.loc[i, 'Drive'] == 4:
            df.loc[i, 'Fase'] = 'loaded_drive_ND'
        # Assegna il valore corrente del contatore delle fasi alla colonna 'Fase'
        else:
            df.loc[i, 'Fase'] = f'fase{phase_counter}'

    return df
'''


def create_phase(df):
    """Crea una nuova colonna 'Fase' nel DataFrame."""
    # Inizializza il contatore delle fasi e la Drive corrente
    phase_counter = 1
    current_drive = df.loc[0, 'Drive']

    # Itera su tutte le righe del dataframe
    for i in range(len(df)):
        # Se la Drive cambia, resetta il contatore delle fasi
        if df.loc[i, 'Drive'] != current_drive:
            phase_counter = 1
            current_drive = df.loc[i, 'Drive']
        # Altrimenti, se lo stimolo cambia e la Drive è tra 5 e 7, incrementa il contatore delle fasi
        elif i > 0 and df.loc[i, 'Stimulus'] != df.loc[i - 1, 'Stimulus'] and df.loc[i, 'Drive'] in [5, 6, 7]:
            phase_counter += 1
        # Se la Drive è tra 0, 1, 2, 3 o 8 assegna il nome specifico della Drive alla colonna 'Fase'
        if df.loc[i, 'Drive'] == 1:
            df.loc[i, 'Fase'] = 'baseline'
        elif df.loc[i, 'Drive'] == 2:
            df.loc[i, 'Fase'] = 'practice_drive'
        elif df.loc[i, 'Drive'] == 3:
            df.loc[i, 'Fase'] = 'relaxing drive'
        elif df.loc[i, 'Drive'] == 4:
            df.loc[i, 'Fase'] = 'loaded_drive_ND'
        elif df.loc[i, 'Drive'] == 8:
            df.loc[i, 'Fase'] = 'failure_drive'
        # Assegna il valore corrente del contatore delle fasi alla colonna 'Fase'
        else:
            df.loc[i, 'Fase'] = f'fase{phase_counter}'

        # Aggiunge l'identificativo della Drive alla colonna 'Fase' per distinguerle meglio
        df.loc[i, 'Fase'] += f'_Drive{df.loc[i, "Drive"]}'

    return df


def unisci_dataframe(elenco_file, cartella):
    """Unisce tutti i DataFrame in un unico DataFrame."""
    lista_dataframe = [create_phase(carica_csv_e_aggiungi_driver(file, cartella)) for file in elenco_file]
    df_totale = pd.concat(lista_dataframe, ignore_index=True)
    return df_totale




# Unisci tutti i DataFrame in un unico DataFrame
df_totale = unisci_dataframe(elenco_file, cartella)

# Salva il DataFrame totale come un nuovo file CSV nella directory corrente
df_totale.to_csv('df_preprocessexp1.csv', index=False)