import numpy as np
import pandas as pd
import scipy.stats as st

import librosa

from tqdm import tqdm


class FeatureGenerator():
    def gen_feat_mfcc(self, X, sub_division=2):
        """ Processa um conjunto de segmentos de sinal acustico.
            (Cada sub-segmento gera 20 coeficientes Mel-cepstrais.)

            Input:
            - X -- dataset do sinal acustico segmentado (numpy.array)
            Output:
            - features -- matriz com os MFCCs de cada segmento colapsados 
                          por linha (numpy.array, shape[n_segmentos, 1500])
            Keywords:
            - sub_division -- indica a subdivisao dos segmentos (max: 3)
        """
        if(sub_division==3): 
            hop_length=301
            n_feats = 60
        elif(sub_division==2): 
            hop_length=751
            n_feats = 40
        else: 
            hop_length=1501
            n_feats = 20
        
        sample_rate = 4e4
        features = np.zeros([len(X), n_feats])

        for i in tqdm(range(len(X))):
            S = librosa.feature.melspectrogram(y = X[i,:], 
                                               sr = sample_rate, 
                                               hop_length = hop_length)
            mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))
            features[i] = mfccs.ravel()
        return features

    def gen_features(self, X):
        """ Gera e retorna todas as features para cada segmento de sinal acustico.

            Input:
            - X -- dataset do sinal acustico segmentado (numpy.array, shape[n_segmentos, 1500]) 
            Output:
            - df_feats -- DataFrame com todas as features geradas para cada segmento
        """

        ## gerando features MFCC
        print('')
        print('Gerando features MFCC:')
        feat_mfcc = self.gen_feat_mfcc(X, sub_division=2)
        cols_mfcc = ['MFCC_'+'{:02d}'.format(n+1) for n in range(feat_mfcc.shape[1])]
        df_feats = pd.DataFrame(feat_mfcc, columns=cols_mfcc)

        ## gerando e adicionando features STAT ao DataFrame
        print('Gerando e adicionando features STAT ao DataFrame:')
        cols_stat = []

        ## Dividindo os segmentos em (INI)cio e (FIM)
        metade_segmento = int(X.shape[1]/2) # 750
        X_ini = X[:, :metade_segmento]
        X_fim = X[:, metade_segmento:]

        ## variancia
        print('\t - Variância ...')
        df_feats['VAR_INI'] = np.var(X_ini, axis=1)
        df_feats['VAR_FIM'] = np.var(X_fim, axis=1)
        cols_stat.append('VAR_INI')
        cols_stat.append('VAR_FIM')

        ## curtose
        print('\t - Curtose ...')
        df_feats['KURT_INI'] = st.kurtosis(X_ini, axis=1)
        df_feats['KURT_FIM'] = st.kurtosis(X_fim, axis=1)
        cols_stat.append('KURT_INI')
        cols_stat.append('KURT_FIM')

        ## quantil
        print('\t - Quantil ...')
        lista_quantis = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        for q in lista_quantis:
            quantil = '{:02d}'.format(q)
            df_feats['Q_'+quantil+'_INI'] = np.quantile(X_ini, float(q)/100, axis=1)
            df_feats['Q_'+quantil+'_FIM'] = np.quantile(X_fim, float(q)/100, axis=1)
            cols_stat.append('Q_'+quantil+'_INI')
            cols_stat.append('Q_'+quantil+'_FIM')

        ## threshold
        print('\t - Threshold ...')
        for th in [5, 10, 20, 40, 80, 160]:
            threshold = '{:03d}'.format(th)
            df_feats['TH_P'+threshold+'_INI'] = np.apply_along_axis(lambda x: np.sum(x >= th), 1, X_ini)
            df_feats['TH_P'+threshold+'_FIM'] = np.apply_along_axis(lambda x: np.sum(x >= th), 1, X_fim)
            df_feats['TH_N'+threshold+'_INI'] = np.apply_along_axis(lambda x: np.sum(x < -th), 1, X_ini)
            df_feats['TH_N'+threshold+'_FIM'] = np.apply_along_axis(lambda x: np.sum(x < -th), 1, X_fim)
            cols_stat.append('TH_P'+threshold+'_INI')
            cols_stat.append('TH_P'+threshold+'_FIM')
            cols_stat.append('TH_N'+threshold+'_INI')
            cols_stat.append('TH_N'+threshold+'_FIM')
        
        return {"df_feats": df_feats,
                "cols_mfcc": cols_mfcc,
                "cols_stat": cols_stat}