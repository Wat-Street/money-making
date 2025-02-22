daily_prices = [
    [45.2, 46.5, 44.8, 45.9, 43.7],  # day 1
    [45.1, 47.2, 46.0, 44.5, 45.8],  # day 2
    [46.3, 45.4, 47.1, 46.8, 45.9],  # day 3
    [46.7, 45.1, 44.9, 46.2, 47.4],  # day 4
    [47.0, 48.3, 46.7, 45.5, 46.9],  # day 5
    [47.5, 46.2, 48.1, 47.3, 47.8],  # day 6
    [48.0, 47.5, 46.9, 48.4, 47.1],  # day 7
    [47.3, 48.6, 47.4, 46.7, 47.2],  # day 8
    [48.1, 49.0, 47.6, 48.4, 48.3],  # day 9
    [48.5, 48.0, 49.2, 47.7, 48.9],  # day 10
    [49.0, 48.8, 49.5, 48.1, 48.7],  # day 11
    [48.9, 49.4, 48.2, 48.0, 49.1],  # day 12
    [49.2, 49.5, 48.4, 49.0, 49.3],  # day 13
    [49.5, 50.1, 48.6, 49.4, 49.8]   # day 14
]

def calculate_volatility(prices):
    if len(prices) < 2:
        return 0

    mean = sum(prices) / len(prices)
    
    squared_diff_sum = sum((price - mean) ** 2 for price in prices)
    
    variance = squared_diff_sum / (len(prices) - 1)
    volatility = variance ** 0.5
    
    return volatility

prime_modulo_5 = []
prime_modulo_3 = []

for i in range(0, 5):
    temp = []
    for jump in range(0, len(daily_prices) // 5):
        print(f'jump: {jump}, i: {i}')
        temp.append(calculate_volatility(daily_prices[jump * 5 + i]))
    prime_modulo_5.append(sum(temp) / len(temp))

for i in range(0, 3):
    temp = []
    for jump in range(0, len(daily_prices) // 3):
        temp.append(calculate_volatility(daily_prices[jump * 3 + i]))
    prime_modulo_3.append(sum(temp) / len(temp))

print("\n")
print("prime_modulo_5: ", prime_modulo_5)
print("prime_modulo_3: ", prime_modulo_3)
print("\n")

prime_term_5 = []
prime_term_3 = []

for i in range(0, 5):
    temp = []
    for j in range(0, len(daily_prices) // 5):
        flat_prices = [price for sublist in daily_prices[j * 5 + i:j * 5 + i + 5] for price in sublist]
        temp.append(calculate_volatility(flat_prices))
    prime_term_5.append(sum(temp) / len(temp))

for i in range(0, 3):
    temp = []
    for j in range(0, len(daily_prices) // 3):
        flat_prices = [price for sublist in daily_prices[j * 3 + i:j * 3 + i + 3] for price in sublist]
        temp.append(calculate_volatility(flat_prices))
    prime_term_3.append(sum(temp) / len(temp))

print("\n")
print("prime_term_5: ", prime_term_5)
print("prime_term_3: ", prime_term_3)
print("\n")
