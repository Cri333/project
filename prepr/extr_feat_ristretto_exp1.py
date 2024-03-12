import pandas as pd
import numpy as np
from datetime import timedelta

# Carica il database
df = pd.read_csv('df_sistema_valori_esperimento1.csv', dtype={14: str})

# Definisci le colonne che vuoi considerare
colonne_da_considerare = ['Failure', 'Palm.EDA', 'Heart.Rate', 'Breathing.Rate',
                          'Perinasal.Perspiration', 'Speed', 'Acceleration', 'Brake', 'Steering']
# Elimina le righe con valori NA solo per le colonne specificate
df.dropna(subset=colonne_da_considerare, inplace=True)

# Crea una nuova colonna 'stimolo_id' che identifica univocamente ogni sequenza di stimoli
df['stimolo_id'] = (df['Stimulus'] != df['Stimulus'].shift()).cumsum()

# Raggruppa i dati per periodo di guida, Drive, stimolo_id, Failure
gruppi = df.groupby(['driver', 'Drive', 'stimolo_id', 'Failure'])

# Calcola le statistiche per ogni gruppo
df_stats = gruppi.agg({
        'Heart.Rate': ['mean'],
        'Time': ['max'],  # Calcola il tempo massimo
        'Palm.EDA': ['mean'],
        'Breathing.Rate': ['mean'],
        'Perinasal.Perspiration': ['mean'],
    })

# Rinomina le colonne
df_stats.columns = ['_'.join(col).strip() for col in df_stats.columns.values]

# Converti il tempo in secondi nel formato "ore.minuti.secondi"
df_stats['Time_max'] = df_stats['Time_max'].apply(lambda x: str(timedelta(seconds=int(x))))

df_stats = df_stats.reset_index().merge(df[['driver', 'Drive', 'stimolo_id', 'Failure', 'Stimulus', 'Fase']].drop_duplicates(),
                                        on=['driver', 'Drive', 'stimolo_id', 'Failure'])

# Rimuovi la colonna stimolo_id e Failure
df_stats.drop(['stimolo_id', 'Failure'], axis=1, inplace=True)

# Definisce l'ordine delle colonne
colonne_ordinate = ['driver', 'Drive', 'Stimulus', 'Fase',
                    'Heart.Rate_mean','Perinasal.Perspiration_mean',
                    'Breathing.Rate_mean',
                    'Time_max',
                    'Palm.EDA_mean']  # Sostituisci con l'ordine delle colonne desiderato

# Riordina le colonne
df_stats = df_stats[colonne_ordinate]
'''calcolo delle formule presenti nel paper'''
# Calcola la media dell'Heart.Rate, Perinasal.Perspiration e Breathing.Rate per la Drive 4 per ogni driver
media_drive_4_hr = df[df['Drive'] == 4].groupby('driver')['Heart.Rate'].mean()
media_drive_4_pp = df[df['Drive'] == 4].groupby('driver')['Perinasal.Perspiration'].mean()
media_drive_4_br = df[df['Drive'] == 4].groupby('driver')['Breathing.Rate'].mean()

# Calcola la media dell'Heart.Rate, Perinasal.Perspiration e Breathing.Rate per le Drive 5, 6, 7 per ogni fase
media_drive_567_hr = df[df['Drive'].isin([5, 6, 7])].groupby(['driver',
                                                              'Drive',
                                                              'Fase'])['Heart.Rate'].mean()
media_drive_567_pp = df[df['Drive'].isin([5, 6, 7])].groupby(['driver',
                                                              'Drive',
                                                              'Fase'])['Perinasal.Perspiration'].mean()
media_drive_567_br = df[df['Drive'].isin([5, 6, 7])].groupby(['driver',
                                                              'Drive',
                                                              'Fase'])['Breathing.Rate'].mean()

def calcola_risultati(media_drive_4, media_drive_567):
    risultati = []
    for idx in media_drive_567.index:
        driver, drive, fase = idx
        if driver in media_drive_4.index:
            risultato = media_drive_567.loc[idx] - media_drive_4.loc[driver]
            risultati.append({'Driver': driver,
                              'Drive': drive,
                              "Fase": fase,
                              "Risultato": risultato})
    return risultati

risultati_hr = calcola_risultati(media_drive_4_hr, media_drive_567_hr)
risultati_pp = calcola_risultati(media_drive_4_pp, media_drive_567_pp)
risultati_br = calcola_risultati(media_drive_4_br, media_drive_567_br)

for res in risultati_hr:
    df_stats.loc[(df_stats['driver'] == res['Driver']) &
                 (df_stats['Drive'] == res['Drive']) &
                 (df_stats['Fase'] == res['Fase']),
                 "Risultato_HR"] = res["Risultato"]

for res in risultati_pp:
    df_stats.loc[(df_stats['driver'] == res['Driver']) &
                 (df_stats['Drive'] == res['Drive']) &
                 (df_stats['Fase'] == res['Fase']),
                 "Risultato_PP"] = res["Risultato"]

for res in risultati_br:
    df_stats.loc[(df_stats['driver'] == res['Driver']) &
                 (df_stats['Drive'] == res['Drive']) &
                 (df_stats['Fase'] == res['Fase']),
                 "Risultato_BR"] = res["Risultato"]

# Salva il DataFrame totale come un nuovo file CSV
df_stats.to_csv('df_extr_feat_ristretto_exp1.csv', index=False)


