import numpy as np
import pandas as pd

# Carica il database
df = pd.read_csv('df_preprocessexp1.csv', dtype={14: str})

drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
                'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']

drive = [4, 5, 6, 7]
# Filtra il DataFrame per includere solo le righe con i driver specificati
df = df[df['driver'].isin(drivers)]
# Filtra il DataFrame per includere solo le righe con le drive specificate
df = df[df['Drive'].isin(drive)]

# Sostituisci i valori nella colonna 'Stimulus'
df['Stimulus'] = df['Stimulus'].replace({2: 1, 4: 5})

'''vado a filtrare i valori in base ai valori corretti descritti nel paper'''
# Sostituisci i valori della frequenza respiratoria che non cadono all'interno dell'intervallo [4, 40] con NaN
df['Breathing.Rate'] = df['Breathing.Rate'].where(df['Breathing.Rate'].between(4, 40), np.nan)
# Sostituisci i valori della frequenza cardiaca che non cadono all'interno dell'intervallo [40, 120] con NaN
df['Heart.Rate'] = df['Heart.Rate'].where(df['Heart.Rate'].between(40, 120), np.nan)
# Sostituisci i valori di Palm.EDA che non cadono all'interno dell'intervallo [28, 628] con NaN
df['Palm.EDA'] = df['Palm.EDA'].where(df['Palm.EDA'].between(28, 628), np.nan)
# Sostituisce tutti i valori di velocità tra -0.1 e 0.1 con zero
df.loc[df['Speed'].between(-0.1, 0.1), 'Speed'] = 0
# Tratta i valori di velocità inferiori a -0.1 come mancanti
df.loc[df['Speed'] < -0.1, 'Speed'] = np.nan
# Sostituisce i valori negativi dell'accelerazione con NaN
df.loc[df['Acceleration'] < 0, 'Acceleration'] = np.nan
# Sostituisce i valori della forza di frenata superiori a 300 con 300
df.loc[df['Brake'] > 300, 'Brake'] = 300

# Aggiungi la nuova colonna 'stress'
df['stress'] = df['Stimulus'].apply(lambda x: 1 if x in [1, 3, 5] else 0)

#funzione per sostituire i valori nulli
def replace_with_mean(x):
    mean = x[(x.notna())].mean()  # Calcola la media escludendo i valori nulli
    x[(x.isna())] = mean  # Sostituisci i valori nulli con la media
    return x

# Applica la funzione alle colonne di interesse
for column in ['Breathing.Rate', 'Perinasal.Perspiration', 'Palm.EDA', 'Heart.Rate', 'Speed', 'Acceleration', 'Steering', 'Brake']:
    df[column] = df.groupby(['driver', 'stress'])[column].transform(replace_with_mean)

# Salva il DataFrame modificato in un nuovo file CSV
df.to_csv('df_sistema_valori_esperimento2.csv', index=False)