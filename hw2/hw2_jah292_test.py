import part_2 as part_2
import sys

if len(sys.argv) > 1:
    part_2.perform_sklearn_backpropagation_neural_network_with_static_word_embeddings_and_pre_processeing(
        'politeness_data.csv',
        sys.argv[1]
    )
else:
    part_2.perform_sklearn_backpropagation_neural_network_with_static_word_embeddings_and_pre_processeing(
        'politeness_data.csv',
        None
    ) 
