import pandas as pd

df = pd.read_csv('./world-data-2023.csv')

# Convert into NL sentence
def generate_sentence(row):
  return (f"{row['Country']} has a population of {row['Population']:} people. " 
            f"The official language is {row['Official language']}, and its capital city is {row['Capital/Major City']}.")

df['Description'] = df.apply(generate_sentence, axis=1)

#print(df[['Country', 'Description']].head())

df_final = df[['Country','Description']]

df_final.to_csv('countries_in_natural_language.csv', index=False)