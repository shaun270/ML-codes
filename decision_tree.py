import math
from collections import Counter

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for cls, cnt in counts.items():
        p = cnt / total
        ent += -p * math.log(p, 2)
    return ent


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    parent_labels = [ex[target_attr] for ex in examples]
    gain = calculate_entropy(parent_labels)

    total = len(examples)
    value_counts = Counter(ex[attr] for ex in examples)

    for v in sorted(value_counts.keys()):  # sorted for deterministic behavior
        subset_labels = [ex[target_attr] for ex in examples if ex[attr] == v]
        gain -= (value_counts[v] / total) * calculate_entropy(subset_labels)

    return gain


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [ex[target_attr] for ex in examples]
    counts = Counter(labels)

    max_count = max(counts.values())
    tied = [cls for cls, c in counts.items() if c == max_count]
    return min(tied)  # alphabetical tie-break


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str):
    """Build a decision tree using the ID3 algorithm (entropy + information gain)."""
    labels = [ex[target_attr] for ex in examples]

    # Base case 1: all same class
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: no attributes left
    if not attributes:
        return majority_class(examples, target_attr)

    # Choose best attribute by IG (tie-break: first in attributes list)
    best_attr = attributes[0]
    best_gain = calculate_information_gain(examples, best_attr, target_attr)

    for attr in attributes[1:]:
        gain = calculate_information_gain(examples, attr, target_attr)
        if gain > best_gain:  # strict > keeps earlier attribute on ties
            best_gain = gain
            best_attr = attr

    tree = {best_attr: {}}

    # Branch in sorted order of values for consistent structure
    values = sorted(set(ex[best_attr] for ex in examples))
    remaining_attrs = [a for a in attributes if a != best_attr]

    for v in values:
        subset = [ex for ex in examples if ex[best_attr] == v]

        # Empty branch -> majority class of current examples
        if not subset:
            tree[best_attr][v] = majority_class(examples, target_attr)
        else:
            tree[best_attr][v] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return tree