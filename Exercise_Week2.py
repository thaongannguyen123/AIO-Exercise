
#%%
import re
#%%
# Tu luan
# cau 1
def find_max_sliding_window(num_list, k):
    result = []
    for i in range(len(num_list) - k + 1):
        max_num = max(num_list[i:i+k])
        result.append(max_num)
    return result

num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3

# Gọi hàm và in kết quả
print(find_max_sliding_window(num_list, k))

# %%
# cau 2
def count_chars(string):
    char_count = {}
    string = string.replace(" ", "").lower()
    for char in string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    return char_count

# example 1
string = 'Happiness'
print(count_chars(string))
# example 2
string = 'smiles'
print(count_chars(string))

# %%
# cau 3
def count_word(file_path):
    word_count = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            words = re.sub(r'[^a-zA-Z\s]', '', line).lower().split()
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    # Sort the dictionary 
    sorted_word_count = dict(sorted(word_count.items()))

    return sorted_word_count

file_path = 'P1_data.txt'
print(count_word(file_path))


# %%
# cau 4
min = __builtins__.min

def levenshtein_distance(source, target):
    # Create a matrix of size (len(source)+1) x (len(target)+1)
    rows = len(source) + 1
    cols = len(target) + 1
    D = [[0] * cols for _ in range(rows)]

    # Initialize the first row and column of the matrix
    for i in range(1, rows):
        D[i][0] = i
    for j in range(1, cols):
        D[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if source[i-1] == target[j-1]:
                cost = 0
            else:
                cost = 1
            D[i][j] = min(
                D[i-1][j] + 1,  # Deletion
                D[i][j-1] + 1,  # Insertion
                D[i-1][j-1] + cost  # Substitution
            )

    return D[-1][-1]

# Example 
source = 'yu'
target = 'you'
distance = levenshtein_distance(source, target)
print(f"The Levenshtein distance between '{source}' and '{target}' is: {distance}")

# %%
# Trac nghiem
# cau 1
max = __builtins__.max

def max_kernel(num_list, k):
    result = []
    for i in range(len(num_list) - k + 1):
        max_num = max(num_list[i:i+k])
        result.append(max_num)
    return result

num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
print(max_kernel(num_list, k))

# %%
# cau 2
def character_count(word):
    character_statistic = {}
    word = word.strip()  
    for char in word:
        if char in character_statistic:
            character_statistic[char] += 1
        else:
            character_statistic[char] = 1
    return character_statistic

print(character_count('smiles'))

# %%
# cau 3
result = count_word(file_path)
assert result['who'] == 3
print(result.get('man', 0))

# %%
# cau 4
assert levenshtein_distance ("hi", " hello ") == 4.0
print(levenshtein_distance("hola", "hello"))

# %%
# cau 5
def check_the_number(N):
    list_of_numbers = []
    result = ""
    for i in range(1, 5):
        list_of_numbers.append(i)
    if N in list_of_numbers:
        result = "True"
    if N not in list_of_numbers:
        result = "False"
    return result

N = 7
assert check_the_number(N) == 'False'
N = 2
results = check_the_number(N)
print(results)

# %%
# cau 6
def my_function(data, max, min):
    result = []
    for i in data:
        if i < min:
            result.append(min)
        elif i > max:
            result.append(max)
        else:
            result.append(i)
    return result

my_list = [10, 2, 5, 0, 1]
max = 2
min = 1
print(my_function(max=max, min=min, data=my_list))

# %%
# cau 7
def my_function(x, y):
    x.extend(y)
    return x

list_num1 = ['a', 2 , 5]
list_num2 = [1 , 1]
list_num3 = [0 , 0]

assert my_function ( list_num1 , my_function ( list_num2 , list_num3 ) ) == ['a', 2 , 5 , 1 , 1 , 0 , 0]

list_num1 = [1, 2]
list_num2 = [3, 4]
list_num3 = [0, 0]

print(my_function(list_num1, my_function(list_num2, list_num3)))

# %%
# cau 8
min = __builtins__.min

def my_function(n):
    return min(n)

my_list = [1 , 22 , 93 , -100]
assert my_function ( my_list ) == -100
my_list = [1, 2, 3, -1]
print(my_function(my_list))

# %%
# cau 9
max = __builtins__.max

def my_function(n):
    return max(n)

my_list = [1001 , 9 , 100 , 0]
assert my_function ( my_list ) == 1001
my_list = [1, 9, 9, 0]
print(my_function(my_list))

# %%
# cau 10
def My_function(integers, number=1):
    return any(x == number for x in integers)

my_list = [1, 3, 9, 4]
assert My_function(my_list , -1) == False

my_list = [1, 2, 3, 4]
print (My_function(my_list, 2))

# %%
# cau 11
def my_function(list_nums=[0, 1, 2]):
    var = 0
    for i in list_nums:
        var += i
    return var / len(list_nums)

print(my_function())

# %%
# cau 12
def my_function(data):
    var = []
    for i in data:
        if i % 3 == 0:
            var.append(i)
    return var

print(my_function([1, 2, 3, 5, 6]))

# %%
# cau 13
def my_function(y):
    var = 1
    while (y > 1):
        # Your code here
        var *= y
        y -= 1
    return var

print(my_function(4))

# %%
# cau 14
def my_function(x):
    return x[::-1]

x = 'apricot'
print(my_function(x))

# %%
# cau 15
def function_helper(x):
    if x > 0:
        return 'T'
    else:
        return 'N'

def my_function(data):
    res = [function_helper(x) for x in data]
    return res

data = [2, 3, 5, -1]
print(my_function(data))

# %%
# cau 16
def function_helper(x, data):
    for i in data:
        if x == i:
            return 0
    return 1

def my_function(data):
    res = []
    for i in data:
        if function_helper(i, res):
            res.append(i)
    return res

lst = [9, 9, 8, 1, 1]
print(my_function(lst))
# %%
