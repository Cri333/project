import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Carica il database
df = pd.read_csv('df_preprocessexp1.csv', dtype={14: str})

# Filtra le righe in base ai valori della colonna 'Drive'
df = df[df['Drive'].isin([4, 5, 6, 7])]

# Sostituisci i valori nella colonna 'Stimulus'
df['Stimulus'] = df['Stimulus'].replace({2: 1, 4: 5})

'''Filtraggio dei valori in base ai valori corretti descritti nel paper'''
# Filtra i valori della frequenza respiratoria che cadono all'interno dell'intervallo [4, 40]
df = df[df['Breathing.Rate'].between(4, 40)]
# Filtra i valori della frequenza cardiaca che cadono all'interno dell'intervallo [40, 120]
df = df[df['Heart.Rate'].between(40, 120)]
# Filtra i valori di Palm.EDA che cadono all'interno dell'intervallo [28, 628]
df = df[df['Palm.EDA'].between(28, 628)]
# Sostituisce tutti i valori di velocità tra -0.1 e 0.1 con zero
df.loc[df['Speed'].between(-0.1, 0.1), 'Speed'] = 0
# Tratta i valori di velocità inferiori a -0.1 come mancanti
df.loc[df['Speed'] < -0.1, 'Speed'] = np.nan
# Sostituisce i valori negativi dell'accelerazione con NaN
df.loc[df['Acceleration'] < 0, 'Acceleration'] = np.nan
# Sostituisce i valori della forza di frenata superiori a 300 con 300
df.loc[df['Brake'] > 300, 'Brake'] = 300

# Definisci le colonne da considerare
colonne_da_considerare = ['Palm.EDA', 'Heart.Rate', 'Breathing.Rate',
        'Perinasal.Perspiration', 'Speed',	'Acceleration',	'Brake', 'Steering']
# Elimina le righe con valori NA solo per le colonne specificate
df.dropna(subset=colonne_da_considerare, inplace=True)

df.to_csv('df_sistema_valori_esperimento1.csv', index=False)
