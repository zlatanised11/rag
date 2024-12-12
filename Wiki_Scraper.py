import wikipediaapi
import pandas as pd
import os
import tqdm

if not os.path.exists('raw_data'):
    os.makedirs('raw_data')

wiki = wikipediaapi.Wikipedia('RAG 2 Riches', 'en', extract_format = wikipediaapi.ExtractFormat.WIKI)

import os
import pandas as pd

def save_processed_csv(wiki, page_title, exclude_sections):
    wiki_page = wiki.page(page_title)
    try:
        assert wiki_page.exists(), print("{} messed up".format(page_title))
    except:pass
    
    data = []

    def process_final_subsections(section, full_title):
        full_title = full_title + ' -> ' + section.title if full_title else section.title
        if section.title not in exclude_sections:
            if not section.sections:
                data.append({
                    'section': full_title,
                    'text': section.text
                })
            else:
                for subsection in section.sections:
                    process_final_subsections(subsection, full_title)

    for s in wiki_page.sections:
        process_final_subsections(s, '')

    df = pd.DataFrame(data, columns=['section', 'text'])
    df.to_csv(os.path.join(os.path.curdir, 'raw_data/{}.csv'.format(page_title)), index=False)

page_names = ['History of Massachusetts','Massachusetts','Massachusetts Bay Colony','History of slavery in Massachusetts'
,'History of education in Massachusetts','History of Springfield, Massachusetts','History of Boston','Outline of Massachusetts'
,'Province of Massachusetts Bay']

exclude_sections = ['See also', 'References', 'Bibliography', 'External links', 'Explanatory notes', 'Further reading']
for page in tqdm.tqdm(page_names):
    save_processed_csv(wiki, page, exclude_sections)

csv_folder = 'raw_data'

dataframes = []

for file in os.listdir(csv_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path)  
        dataframes.append(df)  

combined_df = pd.concat(dataframes, ignore_index=True)

output_file = 'combined_data.csv'
combined_df.to_csv(output_file, index=False)

print(f"All files combined into {output_file}")

input_csv = 'combined_data.csv'
output_md = 'combined_data.md'
df = pd.read_csv(input_csv)

markdown_table = df.to_markdown(index=False, tablefmt="pipe")

with open(output_md, 'w') as md_file:
    md_file.write(markdown_table)

print(f"CSV converted to Markdown and saved to {output_md}")

