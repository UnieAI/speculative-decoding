from register_custom_algorithm import (
    get_custom_ngram_algorithm,
    register_custom_ngram_algorithm,
)

algo = register_custom_ngram_algorithm()
print(f"Algorithm: {algo}")
print(f"is_eagle: {algo.is_eagle()}")
print(f"is_ngram: {algo.is_ngram()}")
print(f"is_none: {algo.is_none()}")
