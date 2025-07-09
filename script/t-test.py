from scipy.stats import ttest_rel


def ttest(my_model, comparison_model):
    t_stat, p_value = ttest_rel(my_model, comparison_model)
    # Interpretation
    if p_value < 0.05:
        print("✅ Statistically significant improvement (p < 0.05)")
    else:
        print("⚠️ No statistically significant difference (p ≥ 0.05)")

    print(f"T-statistic: {t_stat:.5f}")
    print(f"P-value: {p_value:.5f}")


if __name__ == '__main__':
    #  ARGUMENT DETECTION ----------------------------------------------------------------------------------------------
    rooseBERT_scr_uncased_scores = [0.3301656620895921, 0.3302029609345039, 0.33175914531042, 0.3310522707078909,
                                    0.3271505556958863]
    bert_base_uncased_scores = [0.31239310267898, 0.3207886831776821, 0.3235274552855723, 0.3182794771697125,
                                0.316768741394863]
    conflibert_scr_uncased_scores = [0.3076573483215252, 0.3151745224629649, 0.3153311003616306, 0.3031947439117914,
                                     0.318443764040236]

    print(f"Argument Detection {'#' * 100}".upper())
    print("RooseBERT vs BERT:")
    ttest(rooseBERT_scr_uncased_scores, bert_base_uncased_scores)

    print("RooseBERT vs ConfliBERT:")
    ttest(rooseBERT_scr_uncased_scores, conflibert_scr_uncased_scores)

    #  RELATION CLASSIFICATION -----------------------------------------------------------------------------------------
    rooseBERT_scr_cased_scores = [0.6849165662575398, 0.6766931332179259, 0.6656173665791226, 0.667523855458688,
                                  0.6669544455984863]
    bert_base_cased_scores = [0.6529206288369557, 0.6275234486727362, 0.6537824522685857, 0.6538981869188931,
                              0.6501840760525223]
    conflibert_scr_cased_scores = [0.6610098961699417, 0.4959431899658134, 0.6526340260006039, 0.6460737835161665,
                                   0.6463791286128167]

    print(f"Argument Relation Classification {'#' * 100}".upper())
    print("RooseBERT vs BERT:")
    ttest(rooseBERT_scr_cased_scores, bert_base_cased_scores)

    print("RooseBERT vs ConfliBERT:")
    ttest(rooseBERT_scr_cased_scores, conflibert_scr_cased_scores)

    #  NER -------------------------------------------------------------------------------------------------------------
    rooseBERT_scr_cased_scores = [0.4405480822968133, 0.3976779115069365, 0.4153811425314289, 0.4312212661038109,
                                  0.3716378097888492]
    bert_base_cased_scores = [0.4437390042527007, 0.4534863226523645, 0.442892302068622, 0.4553801903386688,
                              0.4430179825416295]
    conflibert_scr_cased_scores = [0.453595279082309, 0.4507577882278386, 0.4715140078646833, 0.447907490576221,
                                   0.4466882054681466]

    print(f"NER {'#' * 100}".upper())
    print("ConfliBERT vs BERT:")
    ttest(conflibert_scr_cased_scores, bert_base_cased_scores)

    print("ConfliBERT vs RooseBERT:")
    ttest(conflibert_scr_cased_scores, rooseBERT_scr_cased_scores)

    #  SENTIMENT ANALYSIS ----------------------------------------------------------------------------------------------
    rooseBERT_scr_uncased_scores = [0.7323561346362649, 0.72661670829864, 0.7225735496288149, 0.7344173441734417,
                                    0.7305455552478538]
    bert_base_uncased_scores = [0.6911793467631527, 0.6911793467631527, 0.6911793467631527, 0.6911793467631527,
                                0.6911793467631527]
    conflibert_scr_uncased_scores = [0.6839609483960949, 0.692706910680142, 0.6938775510204082, 0.697560975609756,
                                     0.6841659610499576]

    print(f"SENTIMENT ANALYSIS {'#' * 100}".upper())
    print("RooseBERT vs BERT:")
    ttest(rooseBERT_scr_uncased_scores, bert_base_uncased_scores)

    print("RooseBERT vs ConfliBERT:")
    ttest(rooseBERT_scr_uncased_scores, conflibert_scr_uncased_scores)
