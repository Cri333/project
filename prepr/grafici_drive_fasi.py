import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv('df_extr_feat_ristretto_exp1.csv', dtype={14: str})

'''# Ottieni l'elenco unico delle Drive
drives = data['Drive'].unique()

# Per ogni Drive
for drive in drives:
    # Filtra i dati per la Drive corrente
    data_drive = data[data['Drive'] == drive]

    # Ottieni l'elenco unico delle Fasi per la Drive corrente
    fasi = data_drive['Fase'].unique()

    # Per ogni Fase
    for fase in fasi:
        # Filtra i dati per la Fase corrente
        data_fase = data_drive[data_drive['Fase'] == fase]

        # Crea un boxplot per la Fase corrente
        plt.figure(figsize=(10, 6))
        data_fase[['Risultato_HR']].boxplot()
        plt.title(f'Drive {drive}, Fase {fase}')
        plt.xlabel('Drive')
        plt.ylabel('Risultato HR')
        plt.show()'''


def crea_boxplots_hr(data, save_dir):
    # Crea la directory se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Ottieni l'elenco unico delle Drive
    drives = data['Drive'].unique()

    # Definisci le proprietà personalizzate per il boxplot
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='black', markersize=3, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # Cambia il colore qui
    whiskerprops = dict(linestyle='--', linewidth=1.5, color='black')
    capprops = dict(linestyle='-', linewidth=1.5, color='black')

    # Per ogni Drive
    for drive in drives:
        # Filtra i dati per la Drive corrente
        data_drive = data[data['Drive'] == drive]

        # Ottieni l'elenco unico delle Fasi per la Drive corrente
        fasi = data_drive['Fase'].unique()

        # Per ogni Fase
        for fase in fasi:
            # Filtra i dati per la Fase corrente
            data_fase = data_drive[data_drive['Fase'] == fase]

            # Crea un boxplot personalizzato per la Fase corrente
            plt.figure()
            data_fase[['Risultato_HR']].boxplot(boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
                                                whiskerprops=whiskerprops, capprops=capprops)

            plt.title(f'Drive {drive}, Fase {fase}')
            plt.xlabel('Drive')
            plt.ylabel('Risultato HR')
            plt.grid(axis='y', linestyle='--')
            #plt.show()
            # Salva il grafico nella directory specificata
            plt.savefig(os.path.join(save_dir, f'Drive_{drive}_Fase_{fase}.png'))

            plt.close()

def crea_boxplots_br(data, save_dir):
    # Crea la directory se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Ottieni l'elenco unico delle Drive
    drives = data['Drive'].unique()

    # Definisci le proprietà personalizzate per il boxplot
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='black', markersize=3, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # Cambia il colore qui
    whiskerprops = dict(linestyle='--', linewidth=1.5, color='black')
    capprops = dict(linestyle='-', linewidth=1.5, color='black')

    # Per ogni Drive
    for drive in drives:
        # Filtra i dati per la Drive corrente
        data_drive = data[data['Drive'] == drive]

        # Ottieni l'elenco unico delle Fasi per la Drive corrente
        fasi = data_drive['Fase'].unique()

        # Per ogni Fase
        for fase in fasi:
            # Filtra i dati per la Fase corrente
            data_fase = data_drive[data_drive['Fase'] == fase]

            # Crea un boxplot personalizzato per la Fase corrente
            plt.figure()
            data_fase[['Risultato_BR']].boxplot(boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
                                                whiskerprops=whiskerprops, capprops=capprops)

            plt.title(f'Drive {drive}, Fase {fase}')
            plt.xlabel('Drive')
            plt.ylabel('Risultato BR')
            plt.grid(axis='y', linestyle='--')
            #plt.show()
            # Salva il grafico nella directory specificata
            plt.savefig(os.path.join(save_dir, f'Drive_{drive}_Fase_{fase}.png'))

            plt.close()


def crea_boxplots_pp(data, save_dir):
    # Crea la directory se non esiste
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Ottieni l'elenco unico delle Drive
    drives = data['Drive'].unique()

    # Definisci le proprietà personalizzate per il boxplot
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    flierprops = dict(marker='o', markerfacecolor='black', markersize=3, linestyle='none')
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # Cambia il colore qui
    whiskerprops = dict(linestyle='--', linewidth=1.5, color='black')
    capprops = dict(linestyle='-', linewidth=1.5, color='black')

    # Per ogni Drive
    for drive in drives:
        # Filtra i dati per la Drive corrente
        data_drive = data[data['Drive'] == drive]

        # Ottieni l'elenco unico delle Fasi per la Drive corrente
        fasi = data_drive['Fase'].unique()

        # Per ogni Fase
        for fase in fasi:
            # Filtra i dati per la Fase corrente
            data_fase = data_drive[data_drive['Fase'] == fase]

            # Crea un boxplot personalizzato per la Fase corrente
            plt.figure()
            data_fase[['Risultato_PP']].boxplot(boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
                                                whiskerprops=whiskerprops, capprops=capprops)

            plt.title(f'Drive {drive}, Fase {fase}')
            plt.xlabel('Drive')
            plt.ylabel('Risultato PP')
            plt.grid(axis='y', linestyle='--')
            #plt.show()
            # Salva il grafico nella directory specificata
            plt.savefig(os.path.join(save_dir, f'Drive_{drive}_Fase_{fase}.png'))

            plt.close()

crea_boxplots_br(data, r'C:\Users\Utente\Desktop\Tirocinio_745297\prepr\grafici_boxplot')
#crea_boxplots_hr(data, r'C:\Users\Utente\Desktop\Tirocinio_745297\prepr\grafici_boxplot')
#crea_boxplots_pp(data, r'C:\Users\Utente\Desktop\Tirocinio_745297\prepr\grafici_boxplot')

