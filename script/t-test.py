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
    #  SENTIMENT ANALYSS  ----------------------------------------------------------------------------------------------
    print(f"Sentiment Analysis".upper())
    best_roosebert = [0.7323561346362649, 0.73021090112298, 0.7300167691447736, 0.7226615762203437, 0.725]
    best_bert = [0.6911793467631527, 0.6911793467631527, 0.6911793467631527, 0.6911793467631527, 0.6836228287841191]
    best_conflibert = [0.6839609483960949, 0.6939799331103679, 0.6964873765093305, 0.6909294512877939,
                       0.6983346983346983]

    print("RooseBERT vs BERT:")
    ttest(best_roosebert, best_bert)

    del best_bert

    print("RooseBERT vs ConfliBERT:")
    ttest(best_roosebert, best_conflibert)

    del best_roosebert
    del best_conflibert
    #  ARGUMENT DETECTION ----------------------------------------------------------------------------------------------
    print(f"Argument Detection".upper())
    best_roosebert = [0.480125293612682, 0.4804703933080648, 0.4798378570599032, 0.4808326567886277, 0.4793058764283013]
    best_bert = [0.4695875544555282, 0.4665518908439708, 0.4656770301817344, 0.4663864303982151, 0.4617379823398828]
    best_conflibert = [0.4672975275177311, 0.4649378841182792, 0.4591647718944465, 0.4633626894671906,
                       0.4656779816369374]

    print("RooseBERT vs BERT:")
    ttest(best_roosebert, best_bert)

    del best_bert

    print("RooseBERT vs ConfliBERT:")
    ttest(best_roosebert, best_conflibert)

    del best_roosebert
    del best_conflibert

    #  RELATION CLASSIFICATION  ----------------------------------------------------------------------------------------------
    print(f"Relation Classification ".upper())
    best_roosebert = [0.6776738338495907, 0.6758947481456635, 0.6690721704710937, 0.6723456739634033,
                      0.6751992255159696]
    best_bert = [0.6653531347701714, 0.6702338433630088, 0.6597793320456513, 0.6526710525974705, 0.6621778828234155]
    best_polibert = [0.6735475620614918, 0.6621611286226843, 0.6665943150331736, 0.6630980991650038, 0.6671327096726647]

    print("RooseBERT vs BERT:")
    ttest(best_roosebert, best_bert)

    del best_bert

    print("RooseBERT vs PoliBERT:")
    ttest(best_roosebert, best_polibert)

    del best_roosebert
    del best_polibert
