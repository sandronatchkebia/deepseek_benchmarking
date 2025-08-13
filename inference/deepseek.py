# -*- coding: utf-8 -*-
import os
import time
import json
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
import random

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
assert API_KEY, "Please set DEEPSEEK_API_KEY in your .env file"

#UPLOAD prompts_metadata.json
#Choose deepseek model
#MODEL_NAME = "deepseek-chat" #V3
MODEL_NAME = "deepseek-reasoner" #R1

# Async OpenAI client
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

GITHUB_PROMPT_BASE = "https://raw.githubusercontent.com/NoamMichael/Comparing-Confidence-in-LLMs/refs/heads/main/Formatted%20Benchmarks"
DATASETS = [
    "boolq_valid",
    "halu_eval_qa",
    "life_eval",
    "lsat_ar_test",
    "sat_en",
    "sciq_test"
]

DATASETS = [
    "sciq_test"
]

SAVE_DIR = "../data/full/deepseek_r1"
#os.makedirs(SAVE_DIR, exist_ok=True)

def download_prompt_csv(dataset_name):
    url = f"{GITHUB_PROMPT_BASE}/{dataset_name}_formatted.csv"
    print(f"⬇️  Downloading {dataset_name}_formatted.csv from GitHub...")
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Failed to download {url}")
    df = pd.read_csv(StringIO(r.text))
    print(f"✅ Downloaded {len(df)} prompts from {dataset_name}")
    return df


def boolq_valid_prompts(df):
    system_prompt = '''
You are a helpful assistant. Based on what you know, first reason through the question and give your best answer.
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct.
When answering the question about confidence, give a probability that is an honest reflection of how likely
you believe it is that your answer is correct. Your response should be in a JSON format. For example:

Question: <Question>

Response: {
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<True or False>",
    "Confidence": "<Probability that your answer is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''
    strip_portion = '''Response: {
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<True or False>",
    "Confidence": "<Probability that your answer is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''

    header = ['Question ID', 'Full Prompt', 'System Prompt']
    output_rows = []
    for i, row in df.iterrows():
        full_prompt = 'Directions:\n' + system_prompt + 'Question: '+ str(row['Question']).capitalize() + '?\n' + 'Response: '
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df


def halu_eval_qa_prompts(df):
    '''
    1. Provide context
    2. Provide Question
    3. Provide real/fake answer rename QID with suffix _r / _h
    4.

    '''
    output_df = pd.DataFrame(columns=['Question ID', 'Full Prompt', 'System Prompt'])
    q_types = ['_r', '_h']
    system_prompt = '''You are a helpful assistant. Based on the context provided, answer the question as accurately as possible.
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct.
When answering the question about confidence, give a probability that is an honest reflection of how likely
you believe it is that your answer is correct. Your response should be in a JSON format.'''

    example = ''' For example:
Context: <Context>
Question: <Question>
Response:
{
    "Answer": "<Your answer>",
    "Confidence": "<The probability that your answer is correct as a float from 0.0 to 1.0>"
}'''
    full_system = system_prompt + example
    for i, row in df.iterrows():
        knowledge = row['knowledge']
        question = row['Question']
        for type in q_types:
            new_row = pd.DataFrame(columns=['Question ID', 'Full Prompt', 'System Prompt'])
            if type == '_r':
                response = row['right_answer']
            else:
                response = row['hallucinated_answer']
            qid = str(row['Question ID']) + type

            full_prompt = f'''Directions: {full_system}\n
Context: {knowledge}\n
Question: {question}\n
Response:
{{
    "Answer": "{response}",
    "Confidence": "'''

            new_row['Question ID'] = pd.Series([qid])
            new_row['Full Prompt'] = pd.Series([full_prompt])
            new_row['System Prompt'] = pd.Series([system_prompt + example])

            output_df = pd.concat([output_df, new_row])
    print(f'HALUEVAL LENGTH: {len(output_df)}')
    return output_df


def life_eval_prompts(df):
    system_prompt = """You are a helpful assistant. Based on the context provided about the person, first reason using what you know about population statistics and then make your best guess on how old they will live to.
Additionally, provide your confidence in the answer as a float from 0.0 to 1.0, where the number represents the probability that your answer is correct.
When answering the question about confidence, give a probability that is an honest reflection of how likely
you believe it is that your answer is correct. Your response should be in a JSON format. For example:

Question: <Question>
Response:
{
    "Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
    "Answer": "<An integer value representing the age you believe the person will die at>",
    "Confidence": "<A float between 0 and 1 representing the probability that the true age at death falls in the range>"
}"""
    header = ['Question ID', 'Full Prompt', 'System Prompt']

    output_rows = []
    for i, row in df.iterrows():
        full_prompt = 'Directions:\n' + system_prompt + '\nQuestion:\n' + row['Question Prompt'] + ' '+ row['Confidence Prompt'] + '\nResponse:\n'
        full_prompt = full_prompt.replace(
            'How certain that your answer is within 1 years of the true value?',
            'How certain are you that your answer is within 1 year of the true value?'
        )
        full_prompt = full_prompt.replace('How certain that your', 'How certain are you that your')
        output_rows.append([row['Question ID'], full_prompt, system_prompt])
    output_df = pd.DataFrame(output_rows, columns=header)
    return output_df


def lsat_ar_test_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words.
Based on your reasoning, Provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>
E) <Option E>

Response:

{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>",
"E": "<Probability choice E is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''

  columns = df.columns
  num_options = columns.str.contains('Option').astype(int).sum()

  #----------------------------------------------------------------------------#
  ## Check if DF is formatted properly
  error_text = f'''Make sure dataframe is in following format:
  | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |

  The current format of Dataframe is: {columns}
  '''
  ['Question ID', 'Question', 'Correct Answer Letter']
  if num_options < 2:
    raise Exception(error_text)

  #----------------------------------------------------------------------------#
  ## Initialize Output dataframe:
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D', 'E']
  options = ['Option A', 'Option B', 'Option C', 'Option D', 'Option E']

  for i in range(len(df)):
    question = df['Question'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    ## Reformat system prompt in order to fit number of options in benchmark
    if type(df['Option E'][i]) == float: ## ABCD
      sys_prompt_temp1 = (sys_prompt1
                    .replace('(A, B, C, D, or E)', '(A, B, C, or D)')
                    .replace('E) ${Option E}', '')
          )

      if type(df['Option D'][i]) == float: ## ABC
        sys_prompt_temp1 = (sys_prompt_temp1
                      .replace('(A, B, C, or D)', '(A, B, or C)')
                      .replace('D) ${Option D}', '')
            )

        if type(df['Option C'][i]) == float: ## AB
          sys_prompt_temp1 = (sys_prompt_temp1
                        .replace('(A, B, or C)', '(A or B)')
                        .replace('C) ${Option C}', '')
              )


    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.split('Response:')[0].replace('<Question>', question).split('For example:')[1].replace('Question:', 'Context:').replace('.Q: ', '.\nQuestion: ')
    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 + (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + '\nResponse:\n'## Specific prompt for question
    prompts1 = prompts1.replace(
            'Question: <Question>',
            '''Context: <Context>
Question: <Question>
''')
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df


def sciq_test_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following question, analyze the options, and provide a concise reasoning for your selected answer. Your reasoning should not exceed 100 words.
Based on your reasoning, provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>

Response:
{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''

  columns = df.columns
  num_options = columns.str_contains('Option').astype(int).sum() if hasattr(columns, 'str_contains') else columns.str.contains('Option').astype(int).sum()

  #----------------------------------------------------------------------------#
  ## Check if DF is formatted properly
  error_text = f'''Make sure dataframe is in following format:
  | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |

  The current format of Dataframe is: {columns}
  '''
  ['Question ID', 'Question', 'Correct Answer Letter']
  if num_options < 2:
    raise Exception(error_text)

  #----------------------------------------------------------------------------#
  ## Initialize Output dataframe:
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D']
  options = ['Option A', 'Option B', 'Option C', 'Option D']

  for i in range(len(df)):
    question = df['Question'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = sys_prompt_temp1.split('Response:')[0].replace('<Question>', question).split('For example:')[1]#.replace('Question:', 'Premise:') ## Uncomment for Qset with premise.
    for j in range(num_options): ## This for loop allows for dynamic question amounts
       new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))

    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 + (new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + '\nResponse:\n'## Specific prompt for question

    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df


def sat_en_prompts(df):  ## Takes in the csv from GitHub and outputs prompts
  ## Takes in a dataframe in the form:
  ## | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  ## |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |
  ##
  ## Returns a dataframe in the form:
  ## | Question ID | Full Prompt 1 | Full Prompt 2 |
  ## |     (Int)       |    (Str)      |    (Str)      |
  sys_prompt1 = '''You are a helpful assistant. Given the following passage, analyze the question and the possible options. Then, provide a concise reasoning for what is the best answer. Your reasoning should not exceed 100 words.
Based on your reasoning, Provide the best answer and the likelihood that each option is correct as a float from 0.0 to 1.0 in a JSON format. The probabilities should sum to 1.0. For example:

Question: <Question>
Options:
A) <Option A>
B) <Option B>
C) <Option C>
D) <Option D>


Response:
{
"Reasoning": "<Concise reasoning for the question. Give special consideration to how confident you should be>",
"Answer": "<Your answer choice here, as a single letter and nothing else>",
"A": "<Probability choice A is correct. As a float from 0.0 to 1.0>",
"B": "<Probability choice B is correct. As a float from 0.0 to 1.0>",
"C": "<Probability choice C is correct. As a float from 0.0 to 1.0>",
"D": "<Probability choice D is correct. As a float from 0.0 to 1.0>"
}

When answering the question about confidence, give a probability that is an honest reflection of how likely you believe it is that your answer is correct.
'''

  columns = df.columns
  num_options = columns.str.contains('Option').astype(int).sum()

  #----------------------------------------------------------------------------#
  ## Check if DF is formatted properly
  error_text = f'''Make sure dataframe is in following format:
  | Question ID | Question | Option A | Option B | ... | Correct Answer Letter |
  |     (Int)       |     (Str)     |  (Str)   |  (Str)   |     |       (Char)          |

  The current format of Dataframe is: {columns}
  '''
  ['Question ID', 'Question', 'Correct Answer Letter']
  if num_options < 2:
    raise Exception(error_text)

  #----------------------------------------------------------------------------#
  ## Initialize Output dataframe:
  header = ['Question ID', 'Full Prompt', 'System Prompt']
  output_df = pd.DataFrame(columns = header)

  #----------------------------------------------------------------------------#

  ## Format questions for benchmark
  letters = ['A', 'B', 'C', 'D']
  options = ['Option A', 'Option B', 'Option C', 'Option D']

  for i in range(len(df)):
    question = df['query'][i].split(' Answer Choices: ')[0]

    sys_prompt_temp1 = sys_prompt1

    option_text = df[options[:num_options]].iloc[i].to_list()
    ## Prompt for specific question
    new_prompt = (sys_prompt_temp1
        .split('Response:')[0]
        .replace('<Question>', question)
        .split('For example:')[1]
        .replace('Question:', 'Passage:')
        .replace('Q: ','\nQuestion: ')
        ) ## Uncomment for Qset with premise.

    for j in range(num_options): ## This for loop allows for dynamic question amounts
        new_prompt = new_prompt.replace(f'<Option {letters[j]}>', str(option_text[j]))


    ## Add formatted prompts.
    ## Note that this is formatted to llama so changes may be needed down the line.
    prompts1 = 'Directions:\n' + sys_prompt1 +(new_prompt.split('<Your concise reasoning here. Max 100 words>')[0])  + 'Response:\n'## Specific prompt for question
    prompts1 = prompts1.replace(
            'Question: <Question>',
            '''Passage: <Passage>
Question: <Question>
''')
    output_df.loc[i] = [df['Question ID'].iloc[i], prompts1,  sys_prompt1]

  return output_df

## Map functions to dataset
functions_map = {
    'boolq_valid': boolq_valid_prompts,
    'halu_eval_qa': halu_eval_qa_prompts,
    'life_eval': life_eval_prompts,
    'lsat_ar_test': lsat_ar_test_prompts,
    'sat_en': sat_en_prompts,
    'sciq_test': sciq_test_prompts
}


def format_prompts(question_set_dict, metadata_path="../prompts_metadata.json"):
    """
    Processes a list of DataFrames to format prompts using metadata.

    Args:
        df_list (list): A list of pandas DataFrames to be processed."
        metadata_path (str): The path to the JSON file containing prompt metadata.

    Returns:
        dict: A dictionary of processed DataFrames with formatted prompts.
    """
    with open(metadata_path, 'r') as f:
        prompts_metadata = json.load(f)
    prompts = {}
    for df_name, df in question_set_dict.items():
        print(f"Processing {df_name} with formatter...")

        format_function = functions_map[df_name]

        prompts_df = format_function(df)

        prompts[df_name] = prompts_df

    print(f'\nTotal Prompt Datasets Formatted: {len(prompts)}')
    print('------------------------------------------\n')
    return prompts


async def _call_with_retry(system_prompt, user_prompt, model, request_timeout=120, max_retries=5, retry_backoff=0.8, semaphore: asyncio.Semaphore | None = None, qid=None):
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            if semaphore:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                        max_tokens=4096,
                        timeout=request_timeout,
                    )
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=4096,
                    timeout=request_timeout,
                )
            return resp
        except Exception as e:
            last_exc = e
            # Respect rate limits / transient errors with exponential backoff + jitter
            sleep_s = (retry_backoff ** attempt) * (1.0 + random.random())
            await asyncio.sleep(min(10.0, max(0.25, sleep_s)))
            attempt += 1
    raise last_exc if last_exc else RuntimeError("Unknown error without exception")


async def run_on_deepseek_async(dataset_name, prompts_df, system_prompt, model=MODEL_NAME, concurrency=16, request_timeout=120):
    results = []
    max_token_hits = 0
    abort_on_token_limit = True  # Set to False if you want to continue despite token limits

    print(f"\n▶ Running: {dataset_name} on {model} ({len(prompts_df)} prompts) [concurrency={concurrency}]\n")

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(prompts_df), desc=f"Processing {dataset_name}")

    async def process_row(i, row):
        nonlocal max_token_hits
        qid = row["Question ID"]
        user_prompt = row["Full Prompt"]
        
        # Check if we should abort due to token limits
        if abort_on_token_limit and max_token_hits > 0:
            return None  # Skip processing if we've hit token limits
            
        try:
            response = await _call_with_retry(system_prompt, user_prompt, model, semaphore=semaphore, qid=qid)
            
            # Check if response hit token limit
            if response.choices[0].finish_reason == "length":
                max_token_hits += 1
                print(f"\n⚠️  QID {qid} hit max_token limit (total: {max_token_hits})")
                
                # If this is the first token limit hit and we want to abort, stop processing
                if abort_on_token_limit and max_token_hits == 1:
                    print(f"\n🚨 Aborting processing due to token limit hit. Saving partial results...")
                    return "ABORT_TOKEN_LIMIT"
            
            results.append({
                "question_id": qid,
                "prompt": user_prompt,
                "response": response.choices[0].message.content,
                "raw_response": response.model_dump()
            })
            return "SUCCESS"
            
        except Exception as e:
            results.append({
                "question_id": qid,
                "error": str(e)
            })
            return "ERROR"
        finally:
            pbar.update(1)

    # Process rows one by one to check for early abort
    for i, row in prompts_df.iterrows():
        result = await process_row(i, row)
        
        # Check if we should abort
        if result == "ABORT_TOKEN_LIMIT":
            break
    
    pbar.close()

    # Determine output filename based on completion status
    if max_token_hits > 0:
        outpath = os.path.join(SAVE_DIR, f"{dataset_name}_{model}_reasoning_results__incomplete.json")
        print(f"\n⚠️  Processing stopped early due to {max_token_hits} token limit hits")
        print(f"   Completed: {len(results)} queries")
        print(f"   Remaining: {len(prompts_df) - len(results)} queries")
    else:
        outpath = os.path.join(SAVE_DIR, f"{dataset_name}_{model}_reasoning_results.json")
        print(f"\n✅ All {len(results)} queries completed successfully")

    # Save results
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: {outpath}")
    
    # Print final summary
    print(f"\n📊 Final Summary:")
    print(f"   Total queries: {len(prompts_df)}")
    print(f"   Successfully completed: {len(results)}")
    print(f"   Hit max_token limit: {max_token_hits}")
    if max_token_hits > 0:
        print(f"   ⚠️  {max_token_hits} queries may have incomplete responses")


async def async_main():
    #Dataset Prep
    processed_datasets = {}
    dataset_row_counts = {} # Dictionary to store dataset name and row count

    for dataset in DATASETS:
        try:
            df_raw = download_prompt_csv(dataset)
            #df_raw = df_raw[:10] #for testing
            # Create a dictionary with the dataset name as the key and the dataframe as the value
            question_set_dict = {dataset: df_raw}
            df = format_prompts(question_set_dict)
            # Access the dataframe for the current dataset from the dictionary
            dataset_df = df[dataset]
            system_prompt = dataset_df["System Prompt"].iloc[0]
            processed_datasets[dataset] = {"prompts_df": dataset_df, "system_prompt": system_prompt}
            dataset_row_counts[dataset] = len(dataset_df) # Store the row count
        except Exception as e:
            print(f"Failed on {dataset}: {e}")

    # Sort the processed_datasets dictionary by row count
    sorted_datasets = sorted(processed_datasets.items(), key=lambda item: dataset_row_counts[item[0]])
    processed_datasets = dict(sorted_datasets)

    print("\nDatasets sorted by row count (increasing order):")
    for dataset, data in processed_datasets.items():
        print(f"- {dataset}: {len(data['prompts_df'])} rows")

    # Optional: small test subset example
    processed_datasets_temp = {}
    if "sat_en" in processed_datasets:
        sat_en_data = processed_datasets["sat_en"]
        processed_datasets_temp["sat_en"] = {
            "prompts_df": sat_en_data["prompts_df"].head(10),
            "system_prompt": sat_en_data["system_prompt"]
        }
    print("Contents of processed_datasets_temp:")
    for dataset, data in processed_datasets_temp.items():
        print(f"- {dataset}: {len(data['prompts_df'])} rows")

    # Inference (sequential across datasets, concurrent within each dataset)
    for dataset, data in processed_datasets.items():
        try:
            await run_on_deepseek_async(
                dataset_name=dataset,
                prompts_df=data["prompts_df"],
                system_prompt=data["system_prompt"]
            )
        except Exception as e:
            print(f"Failed on {dataset}: {e}")


if __name__ == "__main__":
    asyncio.run(async_main())
