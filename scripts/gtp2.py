import os
import ast
import json
import string
import pandas as pd
import numpy as np
import en_core_web_lg
from tqdm import tqdm
from makedataset import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelWithLMHead

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, "Data")


def num_there(s):
    return any(i.isdigit() for i in s)

def add_ner(nlp):
    data = load_dataset()
    data["NER"] = ""
    r_ingredients = []
    for i in tqdm(range(0, len(data))):
        r_ingredients.append(ast.literal_eval(data.ingredients[i]))

    for i in tqdm(range(len(data))):
        ing = []
        for j in range(0, len(r_ingredients[i])):
            doc = nlp(r_ingredients[i][j])
            if len(doc.ents)>0 and num_there(str(doc.ents[0])) == False:
                ing.append(str(doc.ents[0]))
        x = json.dumps(ing)
        data["NER"][i] = x
    data.to_csv(os.path.join(DATA_PATH, "recipes"))

def process_data():
    recipes = pd.read_csv(os.path.join(DATA_PATH, "recipes"))
    recipes.NER = recipes["NER"].apply(lambda row: ast.literal_eval(row))
    recipes = recipes[recipes.NER.apply(lambda row: len(row) > 2)]
    recipes = recipes.reset_index(drop = True)

    #Extract dictionaries
    r_title = recipes.title
    r_recipes = recipes.instructions.apply(lambda row: row.split('\n'))
    r_ingredients = recipes.ingredients.apply(lambda row: ast.literal_eval(row))
    matches = recipes["NER"]

    df = "<RECIPE_START> <INPUT_START> " + matches.str.join(" ") + " <INPUT_END> <INGR_START> " + \
        r_ingredients.str.join(" <NEXT_INGR> ") + " <INGR_END> <INSTR_START> " + \
        r_recipes.str.join(" <NEXT_INSTR> ") + " <INSTR_END> <TITLE_START> " + r_title + " <TITLE_END> <RECIPE_END>"
    train, test = train_test_split(df, test_size=0.1) #use 10% for test set
    np.savetxt(r'unsupervised_train.txt', train, fmt='%s')
    np.savetxt(r'unsupervised_test.txt', test, fmt='%s')
    return recipes

def main():
    nlp = en_core_web_lg.load()
    nlp = nlp.from_disk("./models/v2")
    add_ner(nlp)
    process_data()
    tokenizer = AutoTokenizer.from_pretrained("mbien/recipenlg")
    model = AutoModelWithLMHead.from_pretrained("mbien/recipenlg")
    sequences = "apple milk sugar"
    input = tokenizer.encode(sequences, return_tensors="pt")
    generated = model.generate(input, max_length = 500)

    resulting_string = tokenizer.decode(generated.tolist()[0])

    print("Attempt: ", sequences)
    print("-------------------")
    ing_start = "<INGR_START>"
    ing_end = "<INGR_END>"
    sep = "<NEXT_INGR> "
    title_st = resulting_string.find("<TITLE_START>")
    title_end = resulting_string.find("<TITLE_END>")
    start_idx = resulting_string.find(ing_start)
    end_idx = resulting_string.find(ing_end)
    print("Title:")
    print(resulting_string[title_st+len("<TITLE_START>")+1:title_end].replace(sep, "\n"))
    print("-------------------")
    print("Ingredients:")
    print(resulting_string[start_idx+len(ing_start)+1:end_idx].replace(sep, "\n"))
    print("-------------------")

if __name__ == "main":
    main()