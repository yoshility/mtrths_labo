def get_answer(answer):
    i = -1
    ans = ""
    while (answer[i] != ' '):
        ans = answer[i] + ans
        i -= 1
    return ans

def check_is_correct(gt, pred):
    # put ',' in gt (e.g. 488000 -> 488,000)
    i = len(gt)-1
    cnt = 0
    new_gt = ""
    while i >= 0:
        if cnt != 0 and cnt % 3 == 0:
            new_gt = ',' + new_gt
        new_gt = gt[i] + new_gt
        cnt += 1
        i -= 1
    
    # get final sentence from answer
    new_pred = pred.split('\n\n')[-1]

    return int(new_gt in new_pred or gt in new_pred)