import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pj1_stop_words import get_stop_words


TARGET_SCORE = 0.9
SEARCH_ENGINE = 'd23523bff22df70a8'
JSON_API = 'AIzaSyDPE-QprcLMrjwQAktSOGyatbLrJNWi8Gw'
STOP_WORDS = get_stop_words()

# Alpha, beta, gamma value used in Rocchio's algo
# using common heuristic from https://ieeexplore.ieee.org/document/6591971
ALPHA = 1.0
BETA = 0.75
GAMMA = 0.25


class item:
    '''
    a class used to store the search result
    '''

    def __init__(self, title=None, url=None, description=None, relevant=False):

        self.title = title
        self.url = url
        self.description = description
        self.relevant = relevant


def receive_input(query, search_engine, json_api):
    '''
    Get http response from user input query
    :param query: from user input query
    :param search_engine: search_engin_key
    :param json_api: json api key
    :return: user's response
    '''
    # formate the url
    url = "https://www.googleapis.com/customsearch/v1?key=" + json_api + "&cx=" + search_engine + "&q=" + query
    # send get request via google api.
    # source: https://realpython.com/python-requests/
    resp = requests.get(url)
    return resp


def convert_resp(resp):
    '''
    convert the json response into class item, and calculate the precision
    :param resp: A json response parsed from search result
    :return: A tuple of (list[item], precision)
    '''
    res = []
    i = 0.0
    precision = 0.0
    raw_res = resp.json()
    print('Google Search Results:')
    print('=' * 20)
    for line in raw_res['items']:
        if "mime" in line.keys():
            continue
        i += 1.0
        title = line['title']
        url = line['link']
        summary = line['snippet']
        relevant = get_user_check(i, url, title, summary)
        if relevant:
            precision += 1.0
        items = item(title, url, summary, relevant)
        res.append(items)
    final_precision = precision / i
    return res, final_precision


def get_user_check(i, url, title, summary):
    '''
    :param i: number of i th items in the items list
    :param url: url of an item
    :param title: title of an item
    :param summary: snips of an item
    :return: return ture if the item is relevant, false otherwise
    '''
    print('Result ' + str(int(i)))
    print("[")
    print('URL: ' + url)
    print('TITLE: ' + title)
    print("Summary: " + summary)
    print("]")
    print('Relevant (Y/N)?')
    user_input = input()
    while user_input != "Y" and user_input != "N":
        print("Invalid input, please type Y or N")
        user_input = input()
    if user_input == "Y":
        return True
    return False


def query_expansion(query, res):
    """
    The high-level function to expand the query using tf-idf and Rocchio's algorithm
    :param query: a string of current query
    :param res: a list of class 'item' representing the search results
    :return: a string of expanded query
    """
    # initialize the tf-idf vectorizer
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    document_set = []
    for i in res:
        document_set.append(i.description)
    tfidf_matrix = vectorizer.fit_transform(document_set)
    relevant_doc, irrelevant_doc = get_relevant_doc(res)
    # get vector for both relevant and irrelevant lists of docs
    relevant_vector, irrelevant_vector = get_feats_vector(relevant_doc, irrelevant_doc, vectorizer)

    # get vector for query
    query_vector = vectorizer.transform([query])

    new_query_vector = rocchio_algorithm(relevant_vector, irrelevant_vector, query_vector, len(relevant_doc), len(irrelevant_doc))
    refined_query = refine_query(vectorizer, new_query_vector, query)
    return refined_query


def get_feats_vector(relevant_doc, irrelevant_doc, vectorizer):
    """
    get list of relevant and irrelevant documents
    :param relevant_doc: relevant doc
    :param irrelevant_doc: irrelevant doc
    :param vectorizer: sklearn vectorizer
    :return: relevant, irrelevant doc in vector form
    """
    relevant_vector = get_vector(vectorizer, relevant_doc)
    irrelevant_vector = get_vector(vectorizer, irrelevant_doc)
    return relevant_vector, irrelevant_vector


def get_relevant_doc(res):
    """
    get relevant doc from response
    :param res: google search api result
    :return: relevant, irrelevant doc
    """
    relevant, irrelevant = [], []
    for item in res:
        if item.relevant:
            relevant.append(item.description)
        else:
            irrelevant.append(item.description)
    return relevant, irrelevant


def get_vector(vectorizer, doc):
    """
    get vector form using vectorizer from sklearn
    :param vectorizer: sklearn vectorizer
    :param doc: doc
    :return:
    """
    vector_list = []
    for one in doc:
        vector_list.append(vectorizer.transform([one]))
    # vector = vectorizer.fit_transform(doc)
    return vector_list


def rocchio_algorithm(relevant_vector, irrelevant_vector, query_vector, relevant_len, irrelevant_len):
    """
    Apply rocchio's algorithm on relevant, irrelevant, and query_vectors
    to calculate the new query vector
    :param relevant_vector: relevant doc vector
    :param irrelevant_vector: irrelevant doc vector
    :param query_vector: query in vector form
    :param relevant_len: length of relevant vector
    :param irrelevant_len: length of irrelevant vector
    :return:
    """
    relevant_sum = sum(relevant_vector)
    irrelevant_sum = sum(irrelevant_vector)
    new_vector = ALPHA * query_vector
    new_vector += BETA / relevant_len * relevant_sum
    new_vector -= GAMMA / irrelevant_len * irrelevant_sum
    return new_vector


def refine_query(vectorizer, new_query_vector, query):
    """
    get refined query
    :param vectorizer: sklearn vectorizer
    :param new_query_vector: new query vector using tf-idf weight
    :param query: raw query
    :return: refined query in string type
    """
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(new_query_vector)
    tfidf_vector = tfidf_transformer.transform(new_query_vector)
    feature_names = vectorizer.get_feature_names()

    '''
    convert the tf-idf matrix into a sorted array of words
    reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    https://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
    '''
    sorted_word_arrays = []
    for i in range(tfidf_vector.shape[0]):
        row = tfidf_vector.getrow(i)
        data = row.data
        index = row.indices
        word_scores = {feature_names[index[j]]: data[j] for j in range(len(data))}
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_word_array = [word for word, score in sorted_words]
        sorted_word_arrays.append(sorted_word_array)

    num = 2
    for word in sorted_word_arrays[0]:
        if word in query:
            continue
        num -= 1
        query += ' ' + word
        if num <= 0:
            break
    return query


def main():
    print("Please type the search words: ")
    query = input().lower()
    print("Query: ", query)
    while True:
        resp = receive_input(query, SEARCH_ENGINE, JSON_API)
        if resp.status_code != 200:
            print('request failed, please retry!')
            exit()
        # get items list and current precision
        res, precision = convert_resp(resp)
        print("Precision: ", precision)
        if precision == 0.0:
            print("No result found, terminate")
            exit()
        if precision >= TARGET_SCORE:
            print("Achieved desired precision " + str(TARGET_SCORE) + ", done")
            break
        else:
            print("Below the desired precision 0.9, expanding query...")
            query = query_expansion(query, res)
            query = query.lower()
            print("=" * 20)
            print("Expanded query: ", query)

if __name__ == '__main__':
    main()
