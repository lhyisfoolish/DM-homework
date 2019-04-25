# coding: utf-8
#1. 数据源
#数据集为Wine Reviews中的winemag-data-130k-v2

#2. 要求
#对数据集进行处理，转换成适合关联规则挖掘的形式；
#找出频繁项集；
#导出关联规则，计算其支持度和置信度;
#对规则进行评价，可使用Lift及其它指标, 要求至少2种；
#对挖掘结果进行可视化展示。
# ----------------------------------------------------------

### BEGIN HERE

## Dataset
# Wine Reviews
# https://www.kaggle.com/zynicide/wine-reviews
import pandas as pd
import itertools
import collections
import typing

df = pd.read_csv('./winemag-data-130k-v2.csv',encoding='ISO-8859-1')

# 处理数据
# 选取数据集中的`country`，`designation`，`province`，`variety`，`winery`这5列造transaction。将其中的缺失值用`null`替代

columns = ['country','designation', 'province', 'variety', 'variety', 'winery']
df = df[columns].fillna('null')
transactions = df.values.tolist()
print('%d transactions.' % len(transactions), end='\n\n')

# Apriori算法实现

def join_step(itemsets: typing.List[tuple]):
    i = 0
    while i < len(itemsets):
        skip = 1
        *itemset_first, itemset_last = itemsets[i]
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization
        for j in range(i + 1, len(itemsets)):
            *itemset_n_first, itemset_n_last = itemsets[j]
            if itemset_first == itemset_n_first:
                tail_items_append(itemset_n_last)
                skip += 1
            else:
                break
        itemset_first = tuple(itemset_first)
        for a, b in sorted(itertools.combinations(tail_items, 2)):
            yield itemset_first + (a,) + (b,)
        i += skip


def prune_step(
    itemsets: typing.List[tuple], 
    possible_itemsets: typing.List[tuple]
):
    itemsets = set(itemsets)
    for possible_itemset in possible_itemsets:
        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]
            if removed not in itemsets:
                break
        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def find_itemsets(
    transactions: typing.List[tuple],
    min_support: float,
    max_length: int = 8,
    verbosity: int = 0,
):
    transaction_sets = [set(t) for t in transactions if len(t) > 0]
    transactions = transaction_sets
    use_transaction = collections.defaultdict(lambda: True)

    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    counts = collections.defaultdict(int)
    num_transactions = 0
    for transaction in transactions:
        num_transactions += 1  # Increment counter for transactions
        for item in transaction:
            counts[item] += 1  # Increment counter for single-item itemsets

    large_itemsets = [
        (i, c)
        for (i, c) in counts.items()
        if (c / num_transactions) >= min_support
    ]

    if verbosity > 0:
        num_cand, num_itemsets = len(counts.items()), len(large_itemsets)
        print("  Found {} candidate itemsets of length 1.".format(num_cand))
        print("  Found {} large itemsets of length 1.".format(num_itemsets))

    # If large itemsets were found, convert to dictionary
    if large_itemsets:
        large_itemsets = {1: {(i,): c for (i, c) in sorted(large_itemsets)}}
    else:
        return dict(), num_transactions

    issubset = set.issubset  # Micro-optimization
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))
        itemsets_list = list(large_itemsets[k - 1].keys())
        C_k = list(apriori_gen(itemsets_list))
        C_k_sets = [set(itemset) for itemset in C_k]

        if verbosity > 0:
            print("  Found {} candidate itemsets of length {}.".format(len(C_k), k))
        if not C_k:
            break

        # Prepare counts of candidate itemsets (from the pruen step)
        candidate_itemset_counts = collections.defaultdict(int)
        if verbosity > 1:
            print("    Iterating over transactions.")
        for row, transaction in enumerate(transactions):
            if not use_transaction[row]:
                continue
            found_any = False
            for candidate, candidate_set in zip(C_k, C_k_sets):
                if issubset(candidate_set, transaction):
                    candidate_itemset_counts[candidate] += 1
                    found_any = True
            if not found_any:
                use_transaction[row] = False

        C_k = [
            (i, c)
            for (i, c) in candidate_itemset_counts.items()
            if (c / num_transactions) >= min_support
        ]
        if not C_k:
            break

        large_itemsets[k] = {i: c for (i, c) in sorted(C_k)}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            pp = "  Found {} large itemsets of length {}.".format(num_found, k)
            print(pp)
        k += 1
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")
    return large_itemsets, num_transactions

# 找出支持度大于等于0.06的项集

itemsets, num_trans = find_itemsets(
    transactions=transactions,
    min_support=0.06,
    verbosity=1,
)

# 可视化频繁项集

for k, itemset in itemsets.items():
    print('%s itemsets:' % k)
    for items in itemset:
        print(items)
    print()


# 定义关联规则类`Rule`来表示关联规则。

class Rule(object):
    _decimals = 3

    def __init__(
        self,
        lhs: tuple,
        rhs: tuple,
        count_full: int = 0,
        count_lhs: int = 0,
        count_rhs: int = 0,
        num_transactions: int = 0,
    ):
        self.lhs = lhs  # antecedent
        self.rhs = rhs  # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions

    @property
    def confidence(self):
        try:
            return self.count_full / self.count_lhs
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def support(self):
        try:
            return self.count_full / self.num_transactions
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @staticmethod
    def _pf(s):
        return "{" + ", ".join(str(k) for k in s) + "}"

    def __str__(self):
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))

    def __eq__(self, other):
        return (set(self.lhs) == set(other.lhs)) and (
            set(self.rhs) == set(other.rhs)
        )

    def __hash__(self):
        return hash(frozenset(self.lhs + self.rhs))

    def __len__(self):
        return len(self.lhs + self.rhs)

# 实现Apriori算法

def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_confidence: float,
    num_transactions: int,
    verbosity: int = 0,
):
    def count(itemset):
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")

    for size in itemsets.keys():
        if size < 2:
            continue
        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        for itemset in itemsets[size].keys():
            for removed in itertools.combinations(itemset, 1):
                lhs = set(itemset).difference(set(removed))
                lhs = tuple(sorted(list(lhs)))
                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )

            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(
                itemset, H_1, itemsets, min_confidence, num_transactions
            )

    if verbosity > 0:
        print("Rule generation terminated.\n")


def _ap_genrules(
    itemset: tuple,
    H_m: typing.List[tuple],
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_conf: float,
    num_transactions: int,
):

    def count(itemset):
        return itemsets[len(itemset)][itemset]

    if len(itemset) <= (len(H_m[0]) + 1):
        return

    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    for h_m in H_m:
        lhs = tuple(sorted(list(set(itemset).difference(set(h_m)))))
        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)
    if H_m_copy:
        yield from _ap_genrules(
            itemset, H_m_copy, itemsets, min_conf, num_transactions
        )

# 找出关联规则，设置最小支持度为0.1。

rules = generate_rules_apriori(
    itemsets=itemsets, 
    min_confidence=0.1, 
    num_transactions=num_trans, 
    verbosity=1,
)
rules = list(rules)


# 可视化关联规则

rules_t = []
for rule in rules:
    valid = True
    for val in rule.lhs + rule.rhs:
        if 'null' in val:
            valid = False
            break
    if valid:
        rules_t.append(rule)
rules = rules_t

rules_df = pd.DataFrame({
    'Rules': rules,
    'Left': list(map(lambda x: x.lhs, rules)),    
    'Right': list(map(lambda x: x.rhs, rules)),
    'Support': list(map(lambda x: x.support, rules)),
    'Confidence': list(map(lambda x: x.confidence, rules)),
})
print(rules_df)

#增加Lift指标
def lift(rule):
    observed_support = rule.count_full / rule.num_transactions
    prod_counts = rule.count_lhs * rule.count_rhs
    expected_support = (prod_counts) / rule.num_transactions ** 2
    return observed_support / expected_support
rules_df['Lift'] = list(map(lambda x: lift(x), rules))
print(rules_df)


# 增加Conviction指标
def conviction(self):
    eps = 10e-10  # Avoid zero division
    prob_not_rhs = 1 - self.count_rhs / self.num_transactions
    prob_not_rhs_given_lhs = 1 - self.confidence
    return prob_not_rhs / (prob_not_rhs_given_lhs + eps)

rules_df['Conviction'] = list(map(lambda x: conviction(x), rules))
print(rules_df)

