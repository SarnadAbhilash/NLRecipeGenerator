# # download spacy language model
# !python -m spacy download en_core_web_lg

# import libraries
import en_core_web_lg
import pandas as pd
import re
import os
import random
import spacy
from spacy.util import minibatch, compounding
import warnings
import matplotlib.pyplot as plt
from spacy.training.example import Example

PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, "Data")

# create dictionaries to store the generated food combinations. Do note that one_food != one_worded_food. one_food == "barbecue sauce", one_worded_food == "sauce"
TRAIN_FOOD_DATA = {
    "one_food": [],
    "two_foods": [],
    "three_foods": []
}

TEST_FOOD_DATA = {
    "one_food": [],
    "two_foods": [],
    "three_foods": []
}

# create arrays to store the revision data
TRAIN_REVISION_DATA = []
TEST_REVISION_DATA = []

# create dictionaries to keep count of the different entities
TRAIN_ENTITY_COUNTER = {}
TEST_ENTITY_COUNTER = {}

# This will help distribute the entities (i.e. we don't want 1000 PERSON entities, but only 80 ORG entities)
REVISION_SENTENCE_SOFT_LIMIT = 100

TRAIN_FOOD_DATA_COMBINED = []
TRAIN_DATA = []

def load_data():
    # read in the food csv file
    food_df = pd.read_csv(os.path.join(DATA_PATH, "food.csv"))

    # diaqualify foods with special characters, lowercase and extract results from "description" column
    foods = food_df[food_df["description"].str.contains("[^a-zA-Z ]") == False]["description"].apply(lambda food: food.lower())
    return pre_process(foods)

def pre_process(foods):

    # filter out foods with more than 3 words, drop any duplicates
    foods = foods[foods.str.split().apply(len) <= 3].drop_duplicates()

    # find one-worded, two-worded and three-worded foods
    one_worded_foods = foods[foods.str.split().apply(len) == 1]
    two_worded_foods = foods[foods.str.split().apply(len) == 2]
    three_worded_foods = foods[foods.str.split().apply(len) == 3]

    # total number of foods
    total_num_foods = round(one_worded_foods.size / 45 * 100)

    # shuffle the 2-worded and 3-worded foods since we'll be slicing them
    two_worded_foods = two_worded_foods.sample(frac=1)
    three_worded_foods = three_worded_foods.sample(frac=1)

    # append the foods together 
    foods = one_worded_foods.append(two_worded_foods[:round(total_num_foods * 0.30)]).append(three_worded_foods[:round(total_num_foods * 0.25)])

    return foods

# helper function for deciding what dictionary and subsequent array to append the food sentence on to
def get_food_data(count):
    return {
        1: TRAIN_FOOD_DATA["one_food"] if len(TRAIN_FOOD_DATA["one_food"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["one_food"],
        2: TRAIN_FOOD_DATA["two_foods"] if len(TRAIN_FOOD_DATA["two_foods"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["two_foods"],
        3: TRAIN_FOOD_DATA["three_foods"] if len(TRAIN_FOOD_DATA["three_foods"]) < FOOD_SENTENCE_LIMIT else TEST_FOOD_DATA["three_foods"],
    }[count]

def create_samples(foods):
    food_templates = [
        "I ate my {}",
        "I'm eating a {}",
        "I just ate a {}",
        "I only ate the {}",
        "I'm done eating a {}",
        "I've already eaten a {}",
        "I just finished my {}",
        "When I was having lunch I ate a {}",
        "I had a {} and a {} today",
        "I ate a {} and a {} for lunch",
        "I made a {} and {} for lunch",
        "I ate {} and {}",
        "today I ate a {} and a {} for lunch",
        "I had {} with my husband last night",
        "I brought you some {} on my birthday",
        "I made {} for yesterday's dinner",
        "last night, a {} was sent to me with {}",
        "I had {} yesterday and I'd like to eat it anyway",
        "I ate a couple of {} last night",
        "I had some {} at dinner last night",
        "Last night, I ordered some {}",
        "I made a {} last night",
        "I had a bowl of {} with {} and I wanted to go to the mall today",
        "I brought a basket of {} for breakfast this morning",
        "I had a bowl of {}",
        "I ate a {} with {} in the morning",
        "I made a bowl of {} for my breakfast",
        "There's {} for breakfast in the bowl this morning",
        "This morning, I made a bowl of {}",
        "I decided to have some {} as a little bonus",
        "I decided to enjoy some {}",
        "I've decided to have some {} for dessert",
        "I had a {}, a {} and {} at home",
        "I took a {}, {} and {} on the weekend",
        "I ate a {} with {} and {} just now",
        "Last night, I ate an {} with {} and {}",
        "I tasted some {}, {} and {} at the office",
        "There's a basket of {}, {} and {} that I consumed",
        "I devoured a {}, {} and {}",
        "I've already had a bag of {}, {} and {} from the fridge"
    ]
    # the pattern to replace from the template sentences
    pattern_to_replace = "{}"

    # shuffle the data before starting
    foods = foods.sample(frac=1)

    # the count that helps us decide when to break from the for loop
    food_entity_count = foods.size - 1

    # start the while loop, ensure we don't get an index out of bounds error
    while food_entity_count >= 2:
        entities = []

        # pick a random food template
        sentence = food_templates[random.randint(0, len(food_templates) - 1)]

        # find out how many braces "{}" need to be replaced in the template
        matches = re.findall(pattern_to_replace, sentence)

        # for each brace, replace with a food entity from the shuffled food data
        for match in matches:
            food = foods.iloc[food_entity_count]
            food_entity_count -= 1

            # replace the pattern, but then find the match of the food entity we just inserted
            sentence = sentence.replace(match, food, 1)
            match_span = re.search(food, sentence).span()

            # use that match to find the index positions of the food entity in the sentence, append
            entities.append((match_span[0], match_span[1], "FOOD"))

        # append the sentence and the position of the entities to the correct dictionary and array
        get_food_data(len(matches)).append((sentence, {"entities": entities}))

def load_revision_data(nlp):
    # read in the revision data (just used a random article dataset from a different course I had taken)
    revision_data =  pd.read_csv(os.path.join(DATA_PATH, "npr.csv"))

    return pred_revision(nlp, revision_data)

def load_ner_model():
    return en_core_web_lg.load()


# helper function for incrementing the revision counters
def increment_revision_counters(entity_counter, entities):
    for entity in entities:
        label = entity[2]
        if label in entity_counter:
            entity_counter[label] += 1
        else:
            entity_counter[label] = 1

def convert_revision(nlp, npr_df):
    revision_texts = []

    # convert the articles to spacy objects to better identify the sentences. Disabled unneeded components. # takes ~ 4 minutes
    for doc in nlp.pipe(npr_df["Article"][:6000], batch_size=30, disable=["tagger", "ner"]):
        for sentence in doc.sents:
            if  40 < len(sentence.text) < 80:
                # some of the sentences had excessive whitespace in between words, so we're trimming that
                revision_texts.append(" ".join(re.split("\s+", sentence.text, flags=re.UNICODE)))
    return revision_texts

def pred_revision(nlp, npr_df):
    revision_texts = convert_revision(nlp, npr_df)
    revisions = []
    # Use the existing spaCy model to predict the entities, then append them to revision
    for doc in nlp.pipe(revision_texts, batch_size=50, disable=["tagger", "parser"]):
        
        # don't append sentences that have no entities
        if len(doc.ents) > 0:
            revisions.append((doc.text, {"entities": [(e.start_char, e.end_char, e.label_) for e in doc.ents]}))
    return revisions

def train_test_split(revisions):
    random.shuffle(revisions)
    for revision in revisions:
        # get the entities from the revision sentence
        entities = revision[1]["entities"]

        # simple hack to make sure spaCy entities don't get too one-sided
        should_append_to_train_counter = 0
        for _, _, label in entities:
            if label in TRAIN_ENTITY_COUNTER and TRAIN_ENTITY_COUNTER[label] > REVISION_SENTENCE_SOFT_LIMIT:
                should_append_to_train_counter -= 1
            else:
                should_append_to_train_counter += 1

        # simple switch for deciding whether to append to train data or test data
        if should_append_to_train_counter >= 0:
            TRAIN_REVISION_DATA.append(revision)
            increment_revision_counters(TRAIN_ENTITY_COUNTER, entities)
        else:
            TEST_REVISION_DATA.append(revision)
            increment_revision_counters(TEST_ENTITY_COUNTER, entities)

def train(nlp):
    # combine the food training data
    TRAIN_FOOD_DATA_COMBINED = TRAIN_FOOD_DATA["one_food"] + TRAIN_FOOD_DATA["two_foods"] + TRAIN_FOOD_DATA["three_foods"]
    # join and print the combined length
    TRAIN_DATA = TRAIN_REVISION_DATA + TRAIN_FOOD_DATA_COMBINED
    # add NER to the pipeline and the new label
    ner = nlp.get_pipe("ner")
    ner.add_label("FOOD")

    # get the names of the components we want to disable during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # start the training loop, only training NER
    epochs = 30
    optimizer = nlp.resume_training()
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        sizes = compounding(1.0, 4.0, 1.001)
        
        # batch up the examples using spaCy's minibatc
        for epoch in range(epochs):
            examples = TRAIN_DATA
            random.shuffle(examples)
            batches = minibatch(examples, size=sizes)
            losses = {}
            
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)

            print("Losses ({}/{})".format(epoch + 1, epochs), losses)
    return nlp

def main():
    data = load_data()
    create_samples(data)
    revision_data = load_revision_data()
    train_test_split(revision_data)
    nlp = load_ner_model()
    train(nlp)
    nlp.meta["name"] = "food_entity_extractor_v2"
    nlp.to_disk("./models/v2")

if __name__ == "main":
    main()
