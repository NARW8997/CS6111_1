import sys
import itertools


# Read data from the input file
def read_data(filename):
    with open(filename, 'r') as f:
        transactions = [line.strip().split(',') for line in f.readlines()]
    # skip the first row, which is the item title
    transactions = transactions[1:]
    return transactions


# Step 3: Compute frequent itemsets using the Apriori algorithm
def apriori(transactions, min_sup):
    frequent_itemsets = {}
    k = 1
    k_itemsets, k_max = find_frequent_itemsets(transactions, min_sup)

    while k_itemsets and k <= k_max:
        frequent_itemsets.update(k_itemsets)
        candidates = join_candidates(k_itemsets.keys())
        pruned_candidates = prune(candidates, frequent_itemsets)
        k_itemsets = {candidate: 0 for candidate in pruned_candidates}
        k += 1
        for transaction in transactions:
            for candidate in k_itemsets:
                if candidate.issubset(transaction):
                    k_itemsets[candidate] += 1

        num_transactions = len(transactions)
        k_itemsets = {itemset: count / num_transactions for itemset, count in k_itemsets.items() if
                      count / num_transactions >= min_sup}

    return frequent_itemsets


# Helper function for apriori: Find frequent itemsets
def find_frequent_itemsets(transactions, min_sup):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1

    # prune the initial itemsets
    num_transactions = len(transactions)
    k_max = len(item_counts)
    frequent_itemsets = {frozenset([item]): count / num_transactions for item, count in item_counts.items() if
                         count / num_transactions >= min_sup}
    return frequent_itemsets, k_max


# Helper function for apriori: Get the joint candidates: LHS U RHS
def join_candidates(itemsets):
    candidates = set()
    for itemset1, itemset2 in itertools.combinations(itemsets, 2):
        candidate = itemset1.union(itemset2)
        if len(candidate) == len(itemset1) + 1:
            candidates.add(candidate)
    return candidates


# Helper function for apriori: Prune candidate itemsets
def prune(candidates, prev_frequent_itemsets):
    pruned_candidates = set()
    for candidate in candidates:
        is_valid = True
        for subset in itertools.combinations(candidate, len(candidate) - 1):
            if frozenset(subset) not in prev_frequent_itemsets:
                is_valid = False
                break
        if is_valid:
            pruned_candidates.add(candidate)
    return pruned_candidates


# Step 4: Generate high-confidence association rules
def generate_rules(frequent_itemsets, min_conf):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for item in itemset:
                antecedent = itemset - {item}
                consequent = frozenset([item])
                confidence = support / frequent_itemsets[antecedent]
                if confidence >= min_conf:
                    rules.append((antecedent, consequent, support, confidence))
    return rules


def main():
    # Step 1,2: Accept user prompt as command line args
    filename = sys.argv[1]
    min_sup = float(sys.argv[2])
    min_conf = float(sys.argv[3])

    transactions = read_data(filename)

    # Call apriori algorithm to get the frequent itemsets
    frequent_itemsets = apriori(transactions, min_sup)

    # Generate high confidence association rules
    rules = generate_rules(frequent_itemsets, min_conf)

    # Step 5: Output frequent itemsets to example-run.txt
    with open('output.txt', 'w') as f:
        # Output frequent itemsets
        f.write("==Frequent itemsets (min_sup={}%)\n".format(min_sup * 100))
        for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: -x[1]):
            f.write("[{}], {:.1f}%\n".format(', '.join(itemset), support * 100))

            # Output high-confidence association rules
        f.write("\n==High-confidence association rules (min_conf={}%)\n".format(min_conf * 100))
        for antecedent, consequent, support, confidence in sorted(rules, key=lambda x: -x[3]):
            f.write("[{}] => [{}] (Conf: {:.1f}%, Supp: {:.1f}%)\n".format(
                ', '.join(antecedent), ', '.join(consequent), confidence * 100, support * 100))


if __name__ == '__main__':
    main()
