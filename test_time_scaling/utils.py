def get_answer(dataset_name, answer):
    if dataset_name == 'gsm8k':
        i = -1
        ans = ""
        while (answer[i] != ' '):
            ans = answer[i] + ans
            i -= 1
        return ans
    
    elif dataset_name == 'aime25':
        return str(answer)

def check_is_correct(dataset_name, gt, pred):
    if dataset_name == 'gsm8k':
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
    
    elif dataset_name == 'aime25':
        answer_splited = pred.split('\n')
        return int(gt in answer_splited[-1])