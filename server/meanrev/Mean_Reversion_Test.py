import numpy as np
import math
def get_sma(prices, MA):
    sma = []
    for pos1 in range(0, len(prices) - MA):
        sum = 0
        for change in range(MA, 0, -1):
            sum += prices.iloc[pos1 + change]
        sma.append(sum / MA)
    return sma

def get_sd(prices, time):
    sd = []
    for pos1 in range(0, len(prices) - time):
        sum = 0
        for change in range(time, 0, -1):
            sum += prices.iloc[pos1 + change]
        mean = sum / time
        summa = 0
        for change in range(time, 0, -1):
            summa += (prices.iloc[pos1 + change] - mean) ** 2
        sd.append(math.sqrt(summa / time))
    return sd

def normalize_forward(old_answer):
    answer = old_answer.copy()
    dp_factors = [answer[0]]
    for pos in range(1, len(answer)):
        dp_factors.append(max(abs(answer[pos]), abs(dp_factors[-1])))
        answer[pos] /= dp_factors[pos]
    answer[0] = 1
    return answer

def normalize_average(prices, MAX_HOLDING):
    normalized_prices = [prices[0:int(MAX_HOLDING/2)]]
    for pos in range(MAX_HOLDING/2, len(prices) - MAX_HOLDING/2):
        normalized_prices.append(prices[pos] / (max(prices[pos - MAX_HOLDING/2:pos + MAX_HOLDING/2])))
    return normalized_prices


def test(model, testing_prices, MAX_HOLDING=100):
    try:
        sma10 = get_sma(testing_prices, 10)
        sma30 = get_sma(testing_prices, 20)
        sma50 = get_sma(testing_prices, 50)
        sma100 = get_sma(testing_prices, 100)
        sd10 = get_sd(testing_prices, 10)
        sd30 = get_sd(testing_prices, 20)
        sd50 = get_sd(testing_prices, 50)
        sd100 = get_sd(testing_prices, 100)
        
        MAX_SMA = MAX_HOLDING
        x = []
        for pos in range(0, len(testing_prices) - MAX_SMA):
            x.append([sma10[pos], sma30[pos], sma50[pos], sma100[pos], sd10[pos], sd30[pos], sd50[pos], sd100[pos]])
        x = np.array(x)

        predictions = model.predict(x)
        dates = testing_prices.keys()
        CONFIDENT_X = 10

        bests = [(float(predictions[x][0]), dates[x]) for x in range(CONFIDENT_X)]
        worsts = [(float(predictions[x][0]), dates[x]) for x in range(CONFIDENT_X)]

        for pred_pos in range(len(predictions[CONFIDENT_X:])):
            val = predictions[CONFIDENT_X:][pred_pos]
            min_bests = 100
            max_worsts = -100        
            for pos in range(CONFIDENT_X):
                if bests[pos][0] < min_bests:
                    min_pos = pos
                    min_bests = bests[pos][0]
                if worsts[pos][0] > max_worsts:
                    max_pos = pos
                    max_worsts = worsts[pos][0]
            
            if float(val[0]) > min_bests:
                bests.pop(min_pos)
                bests.append((float(val[0]), dates[pred_pos + CONFIDENT_X]))  # Adjust index for dates
            if float(val[0]) < max_worsts:
                worsts.pop(max_pos)
                worsts.append((float(val[0]), dates[pred_pos + CONFIDENT_X]))  # Adjust index for dates
        
        suggestions = []
        for ele in bests:
            suggestions.append({"datetime": str(ele[1]),
                                "confidence": abs(ele[0]),
                                "suggestion": "Buy"})
        for ele in worsts:
            suggestions.append({"datetime": str(ele[1]),
                                "confidence": abs(ele[0]),
                                "suggestion": "Sell"})
        return suggestions

    except Exception as e:
        print(f"Error in test function: {str(e)}")
        return []  # Return an empty list or handle as needed