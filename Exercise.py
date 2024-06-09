#%%
# libraries
import math
import random

#%%
# Tự Luận
# Câu 1


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
# Câu 3
def calculate_loss(num_samples, loss_name):
    if not num_samples.isnumeric():
        print("Number of samples must be an integer number.")
        return
    
    num_samples = int(num_samples)
    total_loss = 0

    for i in range(num_samples):
        pred = random.uniform(0,10)
        target = random.uniform(0,10)

        if loss_name == 'MAE':
            loss = abs(pred-target)
        elif loss_name == 'MSE':
            loss = (pred-target)**2
        elif loss_name == 'RMSE':
            loss = (pred-target)**2
            total_loss += loss
            print(f'Loss name: {loss_name}, sample: {i}, pred: {pred}, target: {target}, loss: {loss}')
            continue

        print(f'Loss name: {loss_name}, sample: {i}, pred: {pred}, target: {target}, loss: {loss}')
        
    if loss_name == 'RMSE':
        final_rmse = math.sqrt(total_loss/num_samples)
        print(f'final {loss_name}: {final_rmse}')

# test 
num_samples = input('Input number of samples (integer number) which are generated: ')
loss_name = input('Input loss name: ')
calculate_loss(num_samples, loss_name)

#%%
# Câu 4
def factorial(n):
    if n == 0:
        return 1
    else: 
        return n*factorial(n-1)
    
def approx_sin(x, n):
    result = 0
    for i in range(n):
        term = ((-1)**i) * (x**(2*i+1)) / factorial(2*i+1)
        result += term
    return result

def approx_cos(x, n):
    result = 0
    for i in range(n):
        term = ((-1)**i) * (x**(2*i)) / factorial(2*i)
        result += term
    return result

def approx_sinh(x, n):
    result = 0
    for i in range(n):
        term = (x**(2*i+1)) / factorial(2*i+1)
        result += term
    return result

def approx_cosh(x, n):
    result = 0
    for i in range(n):
        term = (x**(2*i)) / factorial(2*i)
        result += term
    return result

# test
approx_cos(x=3.14,n=10)

#%%
# Câu 5
def md_nre_single_sample_cau_5(y, y_hat, n, p):
    
    root_y = y ** (1/n)
    root_y_hat = y_hat ** (1/n)
    
    # Mean Difference of nth Root Error
    md_nre = abs(root_y - root_y_hat) ** p
    
    return md_nre

# examples 
print(md_nre_single_sample_cau_5(y=100, y_hat=99.5, n=2, p=1))  # >> 0.025031328369998107
print(md_nre_single_sample_cau_5(y=50, y_hat=49.5, n=2, p=1))   # >> 0.03544417213033135
print(md_nre_single_sample_cau_5(y=20, y_hat=19.5, n=2, p=1))   # >> 0.05625552183565574
print(md_nre_single_sample_cau_5(y=0.6, y_hat=0.1, n=2, p=1))   # >> 0.45836890322464546

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
def calc_activation_func(x,act_name):
    if act_name == 'sigmoid':
        return 1/(1+math.exp(-x))
    elif act_name == 'relu':
        return max(0,x)
    elif act_name == 'elu':
        alpha = 1
        return x if x > 0 else alpha*(math.exp(x)-1)
    else: raise ValueError("Unsupported activation function")

assert calc_activation_func(x=1,act_name='relu') == 1
print(round(calc_activation_func(x=3,act_name='sigmoid'),2))

# %%
# Câu 7
def calc_ae(y,y_hat):
    return abs(y-y_hat)

y = 1
y_hat = 6
assert calc_ae(y,y_hat) == 5

y = 2
y_hat = 9
print(calc_ae(y,y_hat))

# %%
# Câu 8
def calc_se(y,y_hat):
    return abs(y-y_hat)**2
y = 4
y_hat = 2
assert calc_se(y,y_hat) == 4
print(calc_se(2,1))

# %%
# Câu 9
assert round(approx_cos(x=1, n=10), 2) == 0.54
print(round(approx_cos(x=3.14, n=10), 2))

# %%
# Câu 10
assert round(approx_sin(x=1, n=10), 4) == 0.8415
print(round(approx_sin(x=3.14, n=10), 4))

# %%
# Câu 11
assert round(approx_sinh(x=1,n=10),2) == 1.18
print(round(approx_sinh(x=3.14,n=10),2))

# %%
# Câu 12
assert round(approx_cosh(x=1,n=10),2) == 1.54
print(round(approx_cosh(x=3.14,n=10),2))

# %%
# Câu 13
# (A)
def md_nre_single_sample(y,y_hat,n,p):
    y_root = y ** (1/ n )
    y_hat_root = y_hat ** (1/ n )
    difference = y_root - y_hat_root
    loss = difference ** p
    return loss

# (B)
def md_nre_single_sample1(y,y_hat,n,p):
    y_root = y ** (1/ n )
    y_hat_root = y_hat ** (1/2)
    difference = y_root - y_hat_root
    loss = difference ** p
    return loss

# (C)
def md_nre_single_sample2(y,y_hat,n,p):
    y_root = y ** (1/ n )
    y_hat_root = y_hat ** (1/ n )
    difference = y_root / y_hat_root
    loss = difference ** p
    return loss

# (D)
def md_nre_single_sample3(y,y_hat,n,p):
    y_root = y ** (1/ n )
    y_hat_root = y_hat ** (1/ n )
    difference = y_root - y_hat_root
    loss = difference
    return loss

print(md_nre_single_sample_cau_5(y=100, y_hat=99.5, n=3, p=2))  # >> 6.004561589316014e-05
print(md_nre_single_sample(y=100, y_hat=99.5, n=3, p=2))  # >> 6.004561589316014e-05
print(md_nre_single_sample1(y=100, y_hat=99.5, n=3, p=2))  # >> 28.444940496568623
print(md_nre_single_sample2(y=100, y_hat=99.5, n=3, p=2))  # >> 1.0033472842348656
print(md_nre_single_sample3(y=100, y_hat=99.5, n=3, p=2))  # >> 0.007748910626220962

# %%
