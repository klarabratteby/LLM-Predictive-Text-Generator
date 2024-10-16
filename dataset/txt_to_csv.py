import pandas as pd

# Specify the path of your .txt file and where to save the .csv file
txt_file_path = 'S08_question_answer_pairs.txt'
csv_file_path = './data_questions.csv'

# Load the .txt file as a tab-separated file
df = pd.read_csv(txt_file_path, delimiter='\t')

# Save the DataFrame as a CSV
df.to_csv(csv_file_path, index=False)

print(f"Converted {txt_file_path} to {csv_file_path}")