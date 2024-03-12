import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('df_preprocessexp1.csv', dtype={14: str})



def plot_br(df):
    # Ottieni i driver e i drive unici (tutti)
    '''drivers = df['driver'].unique()
       drives = df['Drive'].unique()'''

    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()


    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Breathing.Rate'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento della Frequenza Respiratoria nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('Frequenza Respiratoria')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()



def plot_hr(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Heart.Rate'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento del battito cardiaco nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('battito cardiaco')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()


def plot_pp(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Perinasal.Perspiration'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento Perinasal Perspiration nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('PP')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()
def plot_palm_EDA(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Palm.EDA'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento del Palm.EDA nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('Palm.EDA')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()

def plot_Speed(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Speed'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento della Velocità nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('Speed')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()

def plot_Accelleration(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Acceleration'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento accelerazione nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('Acceleration')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()


def plot_Brake(df):
    # driver selezionati
    drivers = ['T002', 'T003', 'T005', 'T008', 'T010', 'T014',
               'T016', 'T017', 'T018', 'T020', 'T022', 'T023',
               'T024', 'T029', 'T031', 'T033', 'T034', 'T036',
               'T037', 'T038', 'T039', 'T043', 'T044', 'T045',
               'T047', 'T060', 'T061', 'T066', 'T068', 'T077',
               'T080', 'T081', 'T084']
    drives = df['Drive'].unique()

    # Loop su ogni drive
    for drive in drives:
        # Crea una nuova figura per ogni drive
        plt.figure(figsize=(10, 6))

        # Loop su ogni driver
        for driver in drivers:
            # Filtra i dati per il driver e il drive correnti
            data = df[(df['driver'] == driver) & (df['Drive'] == drive)]

            # Se ci sono dati per questo driver e drive, traccia il grafico
            if not data.empty:
                plt.plot(data['Time'], data['Brake'], label=driver)

        # Aggiungi un titolo e delle etichette al grafico
        plt.title(f'Andamento della forza frenante nel Tempo per Drive {drive}')
        plt.xlabel('Tempo')
        plt.ylabel('Brake')

        # Aggiungi una legenda al grafico
        plt.legend()

        # Mostra il grafico
        plt.show()

def filter_data(df):
    df = df[df['Breathing.Rate'].between(4, 40)]
    df = df[df['Heart.Rate'].between(40, 120)]
    df = df[df['Palm.EDA'].between(28, 628)]
    df.loc[df['Speed'].between(-0.1, 0.1), 'Speed'] = 0
    df.loc[df['Speed'] < -0.1, 'Speed'] = np.nan
    df.loc[df['Acceleration'] < 0, 'Acceleration'] = np.nan
    df.loc[df['Brake'] > 300, 'Brake'] = 300
    return df



plot_br(df)
plot_hr(df)
plot_palm_EDA(df)
plot_pp(df)
plot_Speed(df)
plot_Accelleration(df)
plot_Brake(df)

#Filtra i dati
df_filtered = filter_data(df)

plot_br(df_filtered)
plot_hr(df_filtered)
plot_palm_EDA(df_filtered)
plot_pp(df_filtered)
plot_Speed(df_filtered)
plot_Accelleration(df_filtered)
plot_Brake(df_filtered)

