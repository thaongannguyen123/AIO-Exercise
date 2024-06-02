#%%
# Tự Luận
# Câu 1
import math

def evaluate_classification_model(tp, fp, fn):
    # Check for datatype of tp, fp, fn
    if not isinstance(tp, int) or not isinstance(fp, int) or not isinstance(fn, int):
        raise ValueError("tp, fp, and fn must be integers.")
    
    # Check tp, fp, fn > 0
    if tp <= 0 or fp <= 0 or fn <= 0:
        raise ValueError("tp, fp, and fn must be greater than 0.")
    
    # Calculate Precision, Recall, and F1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

#%%
# Câu 2
def is_number(n):
    try: 
        float(n)
    except ValueError:
        return False
    return True

def activation_function():
    x = input('Input x = ')

    if not is_number(x):
        print('x must be a number.')
        return
    
    x = float(x)

    activation_function_name = input('Input activation function (sigmoid|relu|elu): ')
    if activation_function_name not in ['sigmoid', 'relu', 'elu']:
        print(f'{activation_function_name} is not supported.')
        return
    
    if activation_function_name == 'sigmoid':
        result = 1/(1+math.exp(-x))
    elif activation_function_name == 'relu':
        result = max(0,x)
    elif activation_function_name == 'elu':
        alpha = 0.01
        result = alpha*(math.exp(x)-1) if x <= 0 else x

    print(f'{activation_function_name}: f({x}) = {result}')

activation_function()

#%%


#%%
# Trắc Nghiệm
# Câu 1
assert round(evaluate_classification_model(tp=2, fp=3, fn=5), 2) == 0.33
print(round(evaluate_classification_model(tp=2, fp=4, fn=5), 2))
# >> 0.31

# %%
# Câu 2
assert is_number(3) == 1.0
assert is_number('-2a') == 0.0
print(is_number(1))
print(is_number('n'))
## >> True, False

#%%
# Câu 3
x = -2.0
if x <= 0:
    y = 0.0
else:
    y = x
print(y)
# >> 0.0 => ReLu

# %%
# Câu 4
def calc_sig(x):
    return 1/(1+math.exp(-x))

print(round(calc_sig(2),2))

# %%
# Câu 5
def calc_elu(x):
    alpha = 0.01
    if x <= 0:
        return alpha*(math.exp(x)-1)
    else: 
        return x

print(round(calc_elu(-1),2))

# %%
# Câu 6

# %%
# Câu 7

# %%
# Câu 8

# %%
# Câu 9

# %%
# Câu 10

# %%
# Câu 11

# %%
# Câu 12

# %%
# Câu 13