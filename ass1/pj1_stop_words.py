def get_stop_words():
    with open('stopwords.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    return lines

# def main():
#     get_stop_words()
#
# if __name__ == '__main__':
#     main()