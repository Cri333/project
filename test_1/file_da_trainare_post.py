import pandas as pd


df = pd.read_csv('driver_stress_Time2.csv')
#6 no ND no RD
#8 no PD
#10 no PD RD
#22 no PD RD
#25 no MD
#54 no CD
#61 no PD
#77 no PD
#79 no MD
#84 no PD
'''drivers = ['T002', 'T003', 'T005', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
                'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T066', 'T068',
               'T080', 'T081', 'T006', 'T008', 'T010', 'T022',
            'T025', 'T054', 'T061', 'T077', 'T079', 'T084']'''

'''drivers = ['T002', 'T003', 'T005', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
                'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T066', 'T068',
               'T080', 'T081']

drive = [2, 3, 4, 5, 6, 7]
# Filtra il DataFrame per includere solo le righe con i driver specificati
df = df[df['driver'].isin(drivers)]
# Filtra il DataFrame per includere solo le righe con le drive specificate
df = df[df['Drive'].isin(drive)]'''

# Aggiungi la nuova colonna 'stress'
'''df['stress'] = df['Stimulus'].apply(lambda x: 1 if x in [1, 3, 5] else 0)'''

def replace_with_mean(x):
    mean = x[(x.notna())].mean()  # Calcola la media escludendo i valori nulli
    x[(x.isna())] = mean  # Sostituisci i valori nulli o zero con la media
    return x

# Crea una lista delle colonne da escludere
exclude_columns = ['driver', 'Time_Interval', 'stress']
# Ottieni una lista di tutte le colonne nel DataFrame
all_columns = df.columns.tolist()
# Usa la differenza tra le liste per ottenere le colonne su cui operare
columns_to_include = [col for col in all_columns if col not in exclude_columns]
# Ora columns_to_include può essere incluso nel ciclo for
for column in columns_to_include:
    df[column] = df.groupby(['driver', 'stress'])[column].transform(replace_with_mean)

df = df.dropna(axis=1)



# Salva il DataFrame modificato in un nuovo file CSV
df.to_csv('driver_stress_Time_train2.csv', index=False)
