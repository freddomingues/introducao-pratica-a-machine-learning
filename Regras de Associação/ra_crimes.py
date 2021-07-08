# -*- coding: utf-8 -*-
"""
@author: fred_
"""

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/fclesio/learning-space/master/Datasets/01%20-%20Association%20Rules/Crimes.csv')
df = df.drop(columns=['Crime_id'])

from mlxtend.frequent_patterns import apriori

class Apriori:
    """Apriori Class. Its has Apriori steps."""

    def __init__(self, df, threshold=None, transform_bol=False):
        """Apriori Constructor. 

        :param pandas.DataFrame df: transactions dataset (1 or 0).
        :param float threshold: set threshold for min_support.
        :return: Apriori instance.
        :rtype: Apriori
        """

        self._validate_df(df)

        self.df = df
        if threshold is not None:
            self.threshold = threshold

        if transform_bol:
            self._transform_bol()

    def _validate_df(self, df=None):
        """Validade if df exists. 

        :param pandas.DataFrame df: transactions dataset (1 or 0).
        :return: 
        :rtype: void
        """

        if df is None:
            raise Exception("df must be a valid pandas.DataDrame.")


    def _transform_bol(self):
        """Transform (1 or 0) dataset to (True or False). 

        :return: 
        :rtype: void
        """

        for column in self.df.columns:
            self.df[column] = self.df[column].apply(lambda x: 
                                                    True if (x >= str(3) or
                                                             x == 'Sim' or 
                                                             x == 'Altissimo' or
                                                             x == 'Medio' or
                                                             x == 'Zona_Sul' or
                                                             x == 'Centro' or
                                                             x == 'DP_12_Horas' or
                                                             x == 'Policiamento_via_companhia' or
                                                             x == 'Policia_Comunitaria_Escolar' or
                                                             x == 'Batalhao_Dedicado' or
                                                             x == 'Patrulhamento_Batalhoes_Diversos')
                                                    else False)


    def _apriori(self, use_colnames=False, max_len=None, count=True):
        """Call apriori mlxtend.frequent_patterns function. 

        :param bool use_colnames: Flag to use columns name in final DataFrame.
        :param int max_len: Maximum length of itemsets generated.
        :param bool count: Flag to count length of the itemsets.
        :return: apriori DataFrame.
        :rtype: pandas.DataFrame
        """
    
        apriori_df = apriori(
                    self.df, 
                    min_support=self.threshold,
                    use_colnames=use_colnames, 
                    max_len=max_len
                )
        if count:
            apriori_df['length'] = apriori_df['itemsets'].apply(lambda x: len(x))

        return apriori_df

    def run(self, use_colnames=False, max_len=None, count=True):
        """Apriori Runner Function.

        :param bool use_colnames: Flag to use columns name in final DataFrame.
        :param int max_len: Maximum length of itemsets generated.
        :param bool count: Flag to count length of the itemsets.
        :return: apriori DataFrame.
        :rtype: pandas.DataFrame
        """

        return self._apriori(
                        use_colnames=use_colnames,
                        max_len=max_len,
                        count=count
                    )

    def filter(self, apriori_df, length, threshold):
        """Filter Apriori DataFrame by length and  threshold.

        :param pandas.DataFrame apriori_df: Apriori DataFrame.
        :param int length: Length of itemsets required.
        :param float threshold: Minimum threshold nrequired.
        :return: apriori filtered DataFrame.
        :rtype:pandas.DataFrame
        """
        
        if 'length' not in apriori_df.columns:
            raise Exception("apriori_df has no length. Please run the Apriori with count=True.")

        return apriori_df[ (apriori_df['length'] == length) & (apriori_df['support'] >= threshold) ]

apriori_runner = Apriori(df, threshold=0.9, transform_bol=True)
apriori_df = apriori_runner.run(use_colnames=True)
print(apriori_df)