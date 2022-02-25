# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper 'Generating Diverse
# and Meaningful Captions: Unsupervised Specificity Optimization for Image
# Captioning (Lindh et al., 2018)'
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import os
sys.path.append(os.getcwd())

from neuraltalk2_pytorch import opts
from neuraltalk2_pytorch import train

opt = opts.parse_opt()
print("Launching test run with launch options =", opt)

(test_loss, r1, r5, r10, median_rank, mean_rank,
 avg_score, avg_rel_score_at5, avg_rel_score_at10,
 distinct_captions, novel_captions, vocab_usage,
 evaluation_table_name, current_split,
 least_r1, least_r5, least_r10, least_median_rank, least_mean_rank, least_distinct_captions,
 least_novel_captions, least_vocab_usage,
 most_r1, most_r5, most_r10, most_median_rank, most_mean_rank, most_distinct_captions,
 most_novel_captions, most_vocab_usage,
 lang_stats, least_lang_stats, most_lang_stats,
 avg_lengths, avg_duplicates, avg_lengths_least, avg_duplicates_least, avg_lengths_most, avg_duplicates_most) = train.test(opt=opt, language_eval=True)


# Print SQL table info
print("*** MAIN RESULTS ***")
print("TABLE NAME", evaluation_table_name)
print("CURRENT SPLIT", current_split)
# Print loss function result
print("test_loss", test_loss)
# Print specificity stats
print("r1", r1)
print("r5", r5)
print("r10", r10)
print("median_rank", median_rank)
print("mean_rank", mean_rank)
print("avg_score", avg_score)
print("avg_rel_score_at5", avg_rel_score_at5)
print("avg_rel_score_at10", avg_rel_score_at10)
# Print all diversity stats
print("distinct_captions", distinct_captions)
print("novel_captions", novel_captions)
print("vocab_usage", vocab_usage)
print("lang_stats", lang_stats)
print("avg_lengths", avg_lengths)
print("avg_duplicates", avg_duplicates)

print("*** LEAST SIMILAR, RESULTS ***")
# Print specificity stats
print("least_r1", least_r1)
print("least_r5", least_r5)
print("least_r10", least_r10)
print("least_median_rank", least_median_rank)
print("least_mean_rank", least_mean_rank)
# Print all diversity stats
print("least_distinct_captions", least_distinct_captions)
print("least_novel_captions", least_novel_captions)
print("least_vocab_usage", least_vocab_usage)
print("least_lang_stats", least_lang_stats)
print("avg_lengths_least", avg_lengths_least)
print("avg_duplicates_least", avg_duplicates_least)

print("*** MOST SIMILAR, RESULTS ***")
# Print specificity stats
print("most_r1", most_r1)
print("most_r5", most_r5)
print("most_r10", most_r10)
print("most_median_rank", most_median_rank)
print("most_mean_rank", most_mean_rank)
# Print all diversity stats
print("most_distinct_captions", most_distinct_captions)
print("most_novel_captions", most_novel_captions)
print("most_vocab_usage", most_vocab_usage)
print("most_lang_stats", most_lang_stats)
print("avg_lengths_most", avg_lengths_most)
print("avg_duplicates_most", avg_duplicates_most)
