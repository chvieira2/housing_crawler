
'''
The goal of this module is to train the best model for cold rent sqm prices on the accumulated data over the weeks. This module saves one trained model per week since the begining of the scrapping process.
'''


import time
from housing_crawler.utils import get_data

from config.config import ROOT_DIR

import pandas as pd
import pickle



def train_models():
    """
    This function will train a model per week for all data before a given week number.
    """
    ## Obtain all data
    ads_df = get_data().copy()

    # Transform publication date into timestamp
    ads_df['published_on'] = pd.to_datetime(ads_df['published_on'])

    # Filter ads later than Aug 2022
    ads_df = ads_df[ads_df['published_on'] >= '2022-08-01']

    # Filter ads without a 'price_per_sqm_cold' (nan values)
    ads_df = ads_df.dropna(subset=['price_per_sqm_cold'])

    # Create the week_number tag
    ads_df['week_number'] = ads_df['published_on'].apply(lambda x: x.strftime("%Y")) +'W'+ ads_df['published_on'].apply(lambda x: x.strftime("%V"))

    # Obtain the untrained pipeline with best model. Ideally this sould get it from GitHub but I didn'T manage to make it work with pickle, joblib nor cloudpickle libraries
    prep_pipeline = pickle.load(open(f'{ROOT_DIR}/model/Pipeline_Ridge_untrained.pkl','rb'))

    # prep_pipeline = cloudpickle.load(urlopen("https://github.com/chvieira2/wg_price_predictor/blob/main/wg_price_predictor/models/PredPipeline_WG_allcities_price_per_sqm_cold_untrained.pkl"))
    # # UnpicklingError: invalid load key, '\x0a'.



    # Loop through all weeks and train the model using only data from ads published before that week. Also ignores current week
    for week_number in sorted(list(set(ads_df['week_number']))):

        ## Do not train this weeks model as data hasn't been fully obtained
        if week_number == time.strftime("%Y", time.localtime()) +'W'+ time.strftime("%V", time.localtime()):
            pass
        else:
            ## Check if week's model has been previously trained
            try:
                trained_model = pickle.load(open(f'{ROOT_DIR}/model/trained_models/Pipeline_Ridge_trained_{week_number}.pkl','rb'))
                pass
            except:
                print(f"Pipeline_Ridge_trained_{week_number}.pkl doesn't exist. Creating it.")
                # Identify monday of that week
                monday_week = pd.to_datetime(week_number + '-1', format = "%GW%V-%w")

                # Filter ads older than that week
                ads_df_filtered_week = ads_df[ads_df['published_on'] <= monday_week]

                # Train model
                trained_model = prep_pipeline.fit(ads_df_filtered_week.drop(columns='price_per_sqm_cold'), ads_df_filtered_week['price_per_sqm_cold'])

                # Save (dump) trained model for that week
                with open(f"{ROOT_DIR}/model/trained_models/Pipeline_Ridge_trained_{week_number}.pkl", "wb") as file:
                    pickle.dump(trained_model, file)



if __name__ == "__main__":
     train_models()
