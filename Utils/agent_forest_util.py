from sacrebleu import sentence_bleu

def most_similar_response(resp_list):
    cmp_res = lambda x, y: sentence_bleu(x, [y], lowercase=True).score
    if len(resp_list) == 1:
        return 0, None

    bleu_scores = []
    for idx, agent in enumerate(resp_list):
        total_score = 0
        for idx_o, otheragent in enumerate(resp_list):
            if idx == idx_o:
                continue
            score = cmp_res(agent, otheragent)
            total_score += score
        bleu_scores.append(total_score)

    max_index, max_value = max(enumerate(bleu_scores), key=lambda x: x[1])
    return max_index, bleu_scores
