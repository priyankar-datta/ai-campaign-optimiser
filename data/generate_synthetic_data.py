import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import os

def generate_rows(n=10000, seed=42):
    np.random.seed(seed)
    products = ['Organic Face Cream','Eco Bottle','Smartwatch Pro','NoiseCancel Headphones','Online MBA Course']
    channels = ['facebook','instagram','google_search','youtube']
    audiences = ['Women 25-40','Men 25-40','Students 18-25','Professionals 30-45']
    rows = []
    for i in range(n):
        product = np.random.choice(products)
        channel = np.random.choice(channels)
        audience = np.random.choice(audiences)
        date = (datetime.today() - timedelta(days=np.random.randint(0,90))).strftime('%Y-%m-%d')
        impressions = int(np.random.exponential(1000))
        base_ctr = {
            'facebook':0.01,'instagram':0.012,'google_search':0.03,'youtube':0.008
        }[channel]
        # creative signal
        sentiment = np.random.normal(0.1, 0.3) # arbitrary
        headline_len = np.random.randint(10,60)
        # ctr influenced by device, sentiment, headline length, product
        ctr = max(0.0005, base_ctr + 0.002*sentiment + (headline_len-30)/10000 + np.random.normal(0,0.002))
        clicks = np.random.binomial(impressions, min(0.5,ctr))
        cvr = max(0.005, np.random.beta(2,50)) # conversion rate
        conversions = np.random.binomial(clicks, cvr) if clicks>0 else 0
        spend = impressions * 0.01 * (1 + np.random.rand()*0.5)
        rows.append({
            'ad_id': f'ad_{i}',
            'product': product,
            'channel': channel,
            'audience': audience,
            'date': date,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'spend': round(spend,2),
            'headline_len': headline_len,
            'sentiment': round(sentiment,3)
        })
    df = pd.DataFrame(rows)
    df['ctr'] = df['clicks'] / df['impressions'].replace(0,1)
    df['cvr'] = df['conversions'] / df['clicks'].replace(0,1)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--out', type=str, default='data/synthetic_ads.csv')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_rows(args.rows)
    df.to_csv(args.out, index=False)
    print('Wrote', args.out)

if __name__=='__main__':
    main()
