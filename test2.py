# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 21:34:42 2025

@author: 18236
"""

import pickle
with open('dataset_tvd.pickle', 'rb') as file:
    dataset = pickle.load(file)
dataset = dataset[3000:6000]
from openai import OpenAI

client = OpenAI(api_key="###", base_url="https://api.deepseek.com")


def generate_text(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    
    return response.choices[0].message.content




question_list = "Please output the code for the body of the fopen function in C language code. Just return the code without explanation."
results = generate_text(question_list)



s = 0
# 获取每个函数的代码
for i in range(len(dataset)):
    try:
        s += 1
        print(s)
        code = dataset[i][1]
        prompt = 'You need to determine what computer language this code belongs to, just return the language name. code: ' + code
        code_type = generate_text(prompt)
        fun_names = dataset[i][2]

        temp = {}
        for j in fun_names:
            prompt = 'Please generate the code content of the function body according to the computer language and function name. \Computer language: ' + code_type
            prompt = prompt + '\n' + 'function name: '
            prompt = prompt + j
            fun_code = generate_text(prompt)
            temp[j] = fun_code
        dataset[i].append(code_type)
        dataset[i].append(temp)
        if s%50 == 3:
            with open('dataset_tvd2.pickle', 'wb') as file:
                pickle.dump(dataset, file)  
    except Exception as e:

        print(e)
