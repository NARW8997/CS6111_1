import sys
import re
import openai
import requests
from bs4 import BeautifulSoup
import spacy
from spacy_help_functions import create_entity_pairs
from spanbert import SpanBERT

# simple input:
# python project.py -spanbert <google api key> <google engine id> <openai secret key> 1 0.7 "mark zuckerberg harvard" 10

RELATION = ["Schools_Attended", "Work_For", "Live_in", "Top_Member_Employees"]
BERT_RELATION = {1: "per:schools_attended", 2: "per:employee_of", 3: "per:cities_of_residence",
                 4: "org:top_members/employees"}


# setting up
def print_parameters(google_api_key, google_engine_id, openai_secret_key, method, r, t, k, q):
    """
    print the parameters given by user
    :param google_api_key: google_api_key
    :param google_engine_id: google_engine_id
    :param openai_secret_key: openai_secret_key
    :param method: -spanBert
    :param r: relation given by user
    :param t: a real number t between 0 and 1, indicating the "extraction confidence threshold,"
    :param k: greater than 0, indicating the number of tuples that we request in the output
    :param q: a list of words in double quotes corresponding to a plausible tuple for the relation to extract
    :return: no return, just print all the parameters
    """
    print()
    print('-------- PARAMETERS ---------')
    print("Client Key       = ", google_api_key)
    print("Engine Key       = ", google_engine_id)
    print("OpenAI Key       = ", openai_secret_key)
    print("Method           = ", method)
    print("Relation         = ", RELATION[int(r) - 1])
    print("Threshold        = ", t)
    print("Query            = ", q)
    print("# of Tuples      = ", k)
    print('Loading necessary libraries; This should take a minute or so ...')


# step 2 - get top 10 page search result
def get_top_10(query, search_engine, json_api):
    """
    get top 10 results via google engine
    :param query: query provided
    :param search_engine: engine id
    :param json_api: json api
    :return: return top 10 response
    """
    # formate the url
    url = "https://www.googleapis.com/customsearch/v1?key=" + json_api + "&cx=" + search_engine + "&q=" + query
    # send get request via google api.
    # source: https://realpython.com/python-requests/
    resp = requests.get(url)
    return resp


def retrieve_links(resp, visited_urls):
    """
    Retrieve the links for the top-10 webpage
    :param resp: response from previous steps
    :param visited_urls: urls already visited
    :return: list of all links
    """
    raw_res = resp.json()
    link_list = []
    for item in raw_res['items']:
        link = item['link']
        # skip if the url is already visited
        if link in visited_urls:
            continue
        link_list.append(link)
        visited_urls.append(link)
    return link_list


def extract_from_webpage(link_list, r, t, method, X_dict):
    """
    Extract the actual plain text from the webpage using Beautiful Soup.
    And running via the -spanbert or -gpt3 method
    :param link_list: list of links
    :param r: relations
    :param t: threshold
    :param method: spanbert or gpt3
    :param X_dict: storing all the results tuples
    :return: no return, just need to update X_dict everytime running
    """
    if method != '-spanbert' and method != '-gpt3':
        raise Exception('Provided Method is invalid!')
    nlp = spacy.load("en_core_web_sm")
    count = 0
    for link in link_list:
        count += 1
        response = requests.get(link)
        # only process with response code with 200
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            print("URL ( ", count, " / 10 ): ", link)
            print("\tFetching text from URL...")
            # 3.c truncate the text to 10000 characters
            if len(text) > 10000:
                print("Trimming webpage content from ", len(text), " to 10000 characters")
                text = text[:10000]

            print("\tWebpage Length (number of characters): ", len(text))
            print("\tAnnotating the webpage using SpaCy")
            print()

            doc = nlp(text)
            if method == '-spanbert':
                annotated_sent, relation_number, overall, total_sent = extract_relation_spanbert(doc, r, t, X_dict)
            else:
                annotated_sent, relation_number, overall, total_sent = extract_relation_gpt3(doc, r, X_dict)

            print("Extracted annotations for ", annotated_sent, " out of total", total_sent, "sentences")
            print("Relations extracted from this website: ", relation_number, " (Overall: ", overall, " )")

        if count >= 10:
            break


def sort_X_dict(X_dict):
    """
    sort X_dict according to the confidence score in decreasing order
    :param X_dict: storing all the results tuples
    :return: return sorted dict
    """
    sorted_dict = {k: v for k, v in sorted(X_dict.items(), key=lambda item: item[1][1], reverse=True)}
    return sorted_dict


def convert_X_dict_to_X(X_dict, k, method):
    """
    convert X_dict to set of X
    :param X_dict: storing all the results tuples
    :param k: # of tuples need to convert
    :return: set of X
    """
    X = set()
    for key, value in X_dict.items():
        k -= 1
        if k < 0:
            break
        if method == '-spanbert':
            X.add((key[0], key[1], value[1]))
        else:
            X.add(key)

    return X


def extract_relation_spanbert(doc, r, t, X_dict):
    """
    extract all the relations from spanbert pre-trained model
    :param doc: documents that need process
    :param r: relation
    :param t: threshold
    :param X_dict: storing all the results tuples
    :return: tuple of annotate_sentences, relation_number, relation_number + duplicate, total
    """
    spanbert = SpanBERT("./pretrained_spanbert")
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    count = 0
    total = 0

    for _ in doc.sents:
        total += 1

    print("Extracted ", total,
          " sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
    annotate_sentences = 0
    relation_number = 0
    duplicate = 0

    for sentence in doc.sents:
        count += 1
        if int(count) % 5 == 0:
            print('Processed ' + str(count) + '/' + str(total) + ' sentences')

        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)

        for line in sentence_entity_pairs:

            # if relation is Schools_Attended or Work_For --> only keeping entity pairs with sub: person, obj: organization
            if r == 1 or r == 2:
                if line[1][1] == 'PERSON' and line[2][1] == 'ORGANIZATION':
                    candidate_pairs.append({"tokens": line[0], "subj": line[1], "obj": line[2]})
            # if relation is Live_In --> only keeping entity pairs with sub: person, obj: one of LOCATION, CITY, STATE_OR_PROVINCE, or COUNTRY
            if r == 3:
                if line[1][1] == 'PERSON' and line[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]:
                    candidate_pairs.append({"tokens": line[0], "subj": line[1], "obj": line[2]})
            # if relation is Top_Member_Employees --> only keeping entity pairs with sub: organization, obj: person
            if r == 4:
                if line[1][1] == 'ORGANIZATION' and line[2][1] == 'PERSON':
                    candidate_pairs.append({"tokens": line[0], "subj": line[1], "obj": line[2]})

        if len(candidate_pairs) != 0:
            relation_preds = spanbert.predict(candidate_pairs)
            if len(relation_preds) != 0:
                current_relation_number, current_duplicate, if_annotate = process_relations(X_dict, relation_preds,
                                                                                            candidate_pairs, t, r)
                annotate_sentences += if_annotate
                relation_number += current_relation_number
                duplicate += current_duplicate
    # sorted()

    return annotate_sentences, relation_number, relation_number + duplicate, total


def process_relation_gpt3(X_dict, prompt, relation_number, duplicate, sentence):
    """
    process the sentence from gpt3 model and update X_dict
    :param X_dict: storing all the relation
    :param prompt: prompt using in gpt3
    :param relation_number: number of relation only from this step
    :param duplicate: number of duplicate contains in the step
    :param sentence: sentence used in extraction
    :return: relation_number, duplicate number, and update X_dict
    """
    relations = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        top_p=1,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0
    )

    pattern = r"OBJECT:\s*\"(.*?)\",\s*SUBJECT:\s*\"(.*?)\""
    text = relations['choices'][0]['text']
    matches = re.findall(pattern, text)
    for match in matches:
        subj = match[0]
        obj = match[1]
        print("\t\t=== Extracted Relation ===")
        print("\t\tSentence:   ", sentence)
        print("\t\tSubject: ", subj, " ; Object: ", obj, " ;")
        if (subj, obj) not in X_dict.keys():
            relation_number += 1
            X_dict[(subj, obj)] = ([], 1, False)
            print("\t\tAdding to set of extracted relations")
        else:
            duplicate += 1
            print("\t\tDuplicate. Ignoring this.")
        print("\t\t==========\n")
    return relation_number, duplicate


def extract_relation_gpt3(doc, r, X_dict):
    """
    extract relations using gpt3 model, will call process_relation_gpt3 method
    inside this method
    :param doc: the doc need to be extracted
    :param r: relation
    :param X_dict: storing all the relations
    :return: annotated_sent, relation_number, duplicate + relation_number, total
    """
    total = 0
    annotated_sent, relation_number, duplicate, total_sent = 0, 0, 0, 0
    count = 1

    if r == 1 or r == 2:
        entity_types = ["PERSON", "ORG"]
    elif r == 3:
        entity_types = ["PERSON", "GPE", "LOC"]
    else:
        entity_types = ["ORG", "PERSON"]

    for _ in doc.sents:
        total += 1
        # sent_entities = {ent.label_ for ent in sent.ents}
        # if all(entity_type in sent_entities for entity_type in entity_types):
        #     sentences.append(sent.text)
        #     annotated_sent += 1

    print("Extracted ", total,
          " sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")

    for sent in doc.sents:
        if int(count) % 5 == 0:
            print('Processed ' + str(count) + '/' + str(total) + ' sentences')
        sent_entities = {ent.label_ for ent in sent.ents}
        if all(entity_type in sent_entities for entity_type in entity_types):
            annotated_sent += 1
            sentence = sentence_clean(sent.text)
            if r == 1:
                prompt = (
                    f"Given the sentence: '{sentence}', "
                    "identify all person-school entity pairs representing the 'Schools_Attended' relationship. "
                    "where a person with an actual name is mentioned to have attended a school with an actual name. "
                    "List all pairs in the following format: "
                    "'{{OBJECT: \"person_name\", SUBJECT: \"school_name\"}}'. "
                    "Separate each entity pair with a semicolon (;)."
                    "Example sentence: 'John studied at Harvard University, while Mary went to Stanford University.'\n"
                    "Output: {OBJECT: \"John\", SUBJECT: \"Harvard University\"}; {OBJECT: \"Mary\", SUBJECT: \"Stanford University\"}"
                )
            elif r == 2:
                prompt = (
                    f"Given the sentence: '{sentence}', "
                    "identify all person-organization entity pairs representing the 'Work_For' relationship. "
                    "where a person with an actual name is mentioned to work for an organization or company with an actual name. "
                    "List all pairs in the following format: "
                    "'{{OBJECT: \"person_name\", SUBJECT: \"company_name\"}}'. "
                    "Separate each entity pair with a semicolon (;)."
                    "Example sentence: 'Alice works for Google, and Bob is an employee at Amazon.'\n"
                    "Output: {OBJECT: \"Alice\", SUBJECT: \"Google\"}; {PERSON: \"Bob\", ORGANIZATION: \"Amazon\"}"
                )
            elif r == 3:
                prompt = (
                    f"Given the sentence: '{sentence}', "
                    "identify all person-location entity pairs representing the 'Live_In' relationship. "
                    "List all pairs in the following format: "
                    "where a person with an actual name is mentioned to live in a specific location. "
                    "'{{OBJECT: \"person_name\", SUBJECT: \"location\"}}'. "
                    "Separate each entity pair with a semicolon (;)."
                    "Example sentence: 'Emma lives in New York City, while Jack resides in Los Angeles.'\n"
                    "Output: {OBJECT: \"Emma\", SUBJECT: \"New York City\"}; {OBJECT: \"Jack\", SUBJECT: \"Los "
                    "Angeles\"}"
                )
            else:
                prompt = (
                    f"Given the sentence: '{sentence}', "
                    "identify all organization-person entity pairs representing the 'Top_Member_Employees' "
                     "where a person with name is mentioned as a top member or employee of an organization with name. "
                    "List all pairs in the following format: "
                    "'{{OBJECT: \"organization\", SUBJECT: \"person_name\"}}'. "
                    "Separate each entity pair with a semicolon (;)."
                    "Example sentence: 'Elon Musk is the CEO of SpaceX, and Sundar Pichai is the CEO of Google.'\n"
                    "Output: {OBJECT: \"SpaceX\", SUBJECT: \"Elon Musk\"}; {OBJECT: \"Google\", SUBJECT: \"Sundar Pichai\"}"
                )
            relation_number, duplicate = process_relation_gpt3(X_dict, prompt, relation_number, duplicate, sentence)
        count += 1
    return annotated_sent, relation_number, duplicate + relation_number, total


def process_relations(X_dict, relation_preds, candidate_pairs, t, r):
    """
    process the relations in X_dict, and return relation_number, duplicate, if_annotate
    :param X_dict: storing all the relations
    :param relation_preds: relation preds in previous step
    :param candidate_pairs: candidate paris from previous step
    :param t: threshold
    :param r: relation
    :return: relation_number, duplicate, if_annotate
    """
    relation_number = 0
    duplicate = 0
    if_annotate = 0
    # tuple ([],float, string, string)
    for index in range(len(relation_preds)):
        if relation_preds[index][0] != BERT_RELATION[r]:
            continue
        if relation_preds[index][1] < t:
            continue
        key_tup = (candidate_pairs[index]['subj'][0], candidate_pairs[index]['obj'][0])
        value_tup = (candidate_pairs[index]['tokens'], relation_preds[index][1], False)
        if_annotate = 1
        print("\t\t=== Extracted Relation ===")
        print("\t\tInput Tokens: ", value_tup[0])
        print("\t\tOutput Confidence: ", value_tup[1], " ; Subject: ", key_tup[0], " ; Object: ", key_tup[1], " ; ")
        if key_tup not in X_dict.keys() or X_dict[key_tup][1] < relation_preds[index][1]:
            X_dict[key_tup] = value_tup
            relation_number += 1
            print('\t\tAdding to set of extracted relations')
        # if key_tup is in X_dict and the confidence score >= relation's confidence score
        elif X_dict[key_tup][1] >= relation_preds[index][1]:
            duplicate += 1
            print('\t\tDuplicate with lower confidence than existing record. Ignoring this.')
        else:
            print('else.........................................................')
        print("\t\t==========")
    return relation_number, duplicate, if_annotate


def print_iter_summary_spanbert(X_dict, r, k):
    """
    print the summary from each iteration using spanbert
    :param X_dict: storing all relations
    :param r: relation
    :param k: # of tuples shows in the results
    :return: N/A
    """
    print('================== ALL RELATIONS for', BERT_RELATION[r], ' ( ', min(k, len(X_dict)), ' ) =================')
    for key, value in X_dict.items():
        k -= 1
        if k < 0:
            break
        print('Confidence:', value[1], '           | Subject:', key[0], '           | Object:', key[1])


def print_iter_summary_gpt3(X_dict, r, k):
    """
    print the summary from each iteration using gpt3
    :param X_dict: storing all relations
    :param r: relation
    :param k: # of tuples shows in the results
    :return: N/A
    """
    print('================== ALL RELATIONS for', RELATION[r - 1], ' ( ', min(k, len(X_dict)), ' ) =================')
    for key, value in X_dict.items():
        k -= 1
        if k < 0:
            break
        print('Subject:', key[0], '          | Object:', key[1])


def create_new_query(X_dict):
    """
    create a new query q from tuple y selected from X
    :param X_dict: storing all the relations
    :return: none if no such y tuple exists, otherwise return new query q
    """
    q = None
    for key, value in X_dict.items():
        if value[2] is False:
            q = ' '.join(str(attribute) for attribute in key)
            X_dict[key] = (value[0], value[1], True)
            break
    return q


# Removing redundant newlines and some whitespace characters
def sentence_clean(text):
    """
    make the sentence more pure with raw text, eliminate some symbols,
    no change for text
    :param text: raw text with symbols
    :return: return processed text
    """
    preprocessed_text = re.sub(u'\xa0', ' ', text)

    preprocessed_text = re.sub('\t+', ' ', preprocessed_text)

    preprocessed_text = re.sub('\n+', ' ', preprocessed_text)

    preprocessed_text = re.sub(' +', ' ', preprocessed_text)

    preprocessed_text = preprocessed_text.replace('\u200b', '')

    return preprocessed_text


def main():
    """
    main process to call the step 1 to 6
    :return: return result tuples X
    """
    # if the sys.args are not equal to 0, print error message and exit
    if len(sys.argv) != 9:
        print(
            "Usage: python project.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> "
            "<t> <q> <k>")
        sys.exit(1)
    # Extract the command line arguments
    method = sys.argv[1]
    google_api_key = sys.argv[2]
    google_engine_id = sys.argv[3]
    openai_secret_key = sys.argv[4]
    r = int(sys.argv[5])
    t = float(sys.argv[6])
    q = sys.argv[7]
    k = int(sys.argv[8])
    # assign openai api key
    openai.api_key = openai_secret_key
    # initialize visited_urls list used in retrieve_links
    visited_urls = []
    # print parameters
    print_parameters(google_api_key, google_engine_id, openai_secret_key, method, r, t, k, q)
    # initialize number of iterations and total relations number
    iter_num = 0
    relations_num = 0
    # initialize X_dict to store all relations with their sentence, confidence and used vars
    X_dict = {}

    # step 2 to 6
    # if total relations_num less than k, then loop the process
    while relations_num < k:
        print('=========== Iteration:', iter_num, ' - Query:', q, '===========')
        # get top 10 response
        resp = get_top_10(q, google_engine_id, google_api_key)
        # get list of link without duplicate
        link_list = retrieve_links(resp, visited_urls)
        # extract from webpage and update X_dict
        extract_from_webpage(link_list, r, t, method, X_dict)
        # sort X_dict according to the confidence in decreasing order
        X_dict = sort_X_dict(X_dict)
        # reassign relations_num
        relations_num = len(X_dict)
        # If X contains at least k tuples, return the top-k such tuples and stop
        # otherwise, create a query q
        if relations_num < k:
            q = create_new_query(X_dict)
            # If no such y tuple exists, then stop
            # if no such q exist, then break the loop and return
            if q is None:
                break
        # update iteration number within each iter
        iter_num += 1
        # print summary of each iteration
        if method == '-spanbert':
            print_iter_summary_spanbert(X_dict, r, k)
        else:
            print_iter_summary_gpt3(X_dict, r, k)

    # print total # of iterations
    print('Total # of iterations =', iter_num)
    # return the set of X
    return convert_X_dict_to_X(X_dict, k, method)


if __name__ == '__main__':
    main()
