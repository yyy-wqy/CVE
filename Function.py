# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:56:24 2025

@author: hanly2
"""
import re
import requests
import zipfile
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pickle
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import string
from collections import Counter

import os
import ast




def generate_prompt(tvd):
    with open('prompt.txt','r') as f:
        ff = f.read()
    prompt = ff +"\n"+ "\"" + tvd + "\"." + "\n I always extract Affected Product and Version from the original text according to the order of their appearance."
    prompt = prompt + "\n}\n" + "The following are the results (in English) extracted from the original text: "
    return prompt

def regular(text): 
    pattern = r'(\d+)\. "(.*?)" is the (.*?)\.'
    matches = re.findall(pattern, text)
    
    # 构建字典
    result = {}
    for match in matches:
        result[match[2]] = match[1]
    r=[text, result]
    return r

def download(name, url, path):
    repo_url = url

    zip_response = requests.get(repo_url)

    if zip_response.status_code == 200:
        # 将压缩包保存为文件
        with open(f"{path}/{name}.zip", "wb") as f:
            f.write(zip_response.content)
        
        # 解压
        with zipfile.ZipFile(BytesIO(zip_response.content)) as zip_ref:
            zip_ref.extractall(f"{path}/{name}")

    else:
        print(f"Error: Failed to download the source code for {name}")
    
def fetch_dynamic_content(url):
    # 设置 Chrome 浏览器选项
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    # 自动安装匹配的 ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        page_content = driver.page_source
        return page_content
    finally:
        driver.quit()

def get_dependences(url):
    content = fetch_dynamic_content(url)
    with open("D:\CCS2025\coding\dynamic_content.html", "w", encoding="utf-8") as file:
        file.write(content)

    from bs4 import BeautifulSoup
    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(content, 'html.parser')

    # 找到所有依赖项的表格行
    dependencies = soup.find_all('tr', class_='hhy1BTUUKFxBQX2kJPTJ')

    # 提取每个依赖的名称和版本
    dependencies_list = []
    for dep in dependencies:
        package = dep.find('a')
        version = dep.find('div', class_='SVYxipenVB4gnQt4bIe8')
        if package and version:
            dependencies_list.append([package.text.strip(), version.text.strip()])
    return dependencies_list


def get_tvds_based_sbom(sbom):
    with open(r'all_cve.data','rb') as f:
        data_all_cve = pickle.load(f)
    cve_des = []
    for i in data_all_cve:
        cve_des.append(data_all_cve[i][0])
    tvds_sbom = {}
    for i in sbom:
        product = i[0]
        product_version = i[1]
        temp = []
        for j in cve_des:
            if product in j[0].lower():# or product_version in j.low():
                temp.append(j)
        tvds_sbom['-'.join(i)] = temp
    return tvds_sbom

def generate_prompt_rootcause_impact(tvd):
    with open('prompt_rootcause_imapct.txt','r') as f:
        ff = f.read()
    prompt = ff +"\n"+ "\"" + tvd + "\"." + "\n I always extract 2 aspects from the original text according to the order of their appearance, namely Root Cause and Impact. However, each of these aspects may not be present, and their order is not fixed."
    prompt = prompt + "\n}\n" + "The following are the results (in English) extracted from the original text: "
    return prompt





def LDA_keyaspect(groups, num_topics):
    nltk.download('stopwords')
    
    # 停用词列表
    stop_words = set(stopwords.words('english'))
    
    # 数据预处理函数

    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        return tokens    
    # 示例数据
    # group1 = ["I love machine learning and artificial intelligence.", "Deep learning is a subset of machine learning."] * 500
    # group2 = ["Natural language processing is a field of AI.", "Artificial intelligence is transforming the world."] * 50
    # group3 = ["Data science involves statistics, machine learning, and data analysis.", "Machine learning is everywhere."] * 300
    
    # 所有分组数据
    # groups = [group1, group2, group3]
    
    # 预处理每组数据
    processed_groups = [list(map(preprocess_text, group)) for group in groups]
    
    # 构建字典
    all_texts = [doc for group in processed_groups for doc in group]
    dictionary = corpora.Dictionary(all_texts)
    
    # 创建词袋模型
    corpus_groups = [[dictionary.doc2bow(doc) for doc in group] for group in processed_groups]
    
    # 获取每个组的文档数量
    group_sizes = [len(group) for group in corpus_groups]
    
    # 对每组的词袋模型进行归一化
    normalized_corpora = []
    for corpus, size in zip(corpus_groups, group_sizes):
        scaling_factor = 1 / size  # 归一化因子
        normalized_corpus = [
            [(term_id, count * scaling_factor) for term_id, count in doc] for doc in corpus
        ]
        normalized_corpora.append(normalized_corpus)
    
    # 合并所有组的归一化语料
    combined_corpus = [doc for corpus in normalized_corpora for doc in corpus]
    
    # 训练 LDA 模型
    # num_topics = 5  # 设置主题数量
    lda_model = LdaModel(corpus=combined_corpus, id2word=dictionary, num_topics=num_topics, passes=15)
    
    # 获取每个 group 的主题分布
    group_topic_distributions = []
    for corpus in corpus_groups:
        group_topics = []
        for doc in corpus:
            topic_distribution = lda_model.get_document_topics(doc)
            top_topic = max(topic_distribution, key=lambda x: x[1])[0]  # 获取概率最高的主题
            group_topics.append(top_topic)
        group_topic_distributions.append(group_topics)
    
    # 跨 group 统计主题频次
    group_common_topics = []
    for group_topics in group_topic_distributions:
        group_common_topics.extend(group_topics)
    
    # 统计每个主题在所有 group 中的频次
    topic_counter = Counter(group_common_topics)
    most_common_topic = topic_counter.most_common(1)[0]  # 出现最多的主题
    
    # 输出结果
    print("主题关键词:")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"主题 {idx}: {topic}")
    
    print("\n每个组的主题分布:")
    for group_idx, group_topics in enumerate(group_topic_distributions):
        print(f"组 {group_idx + 1}: {Counter(group_topics)}")
    
    print(f"\n跨组出现最多的主题是主题 {most_common_topic[0]}，出现次数为 {most_common_topic[1]} 次。")


def LDA_keyaspect1(groups, num_topics):
    import nltk
    import string
    import numpy as np
    from sklearn.cluster import KMeans
    from gensim.models import KeyedVectors
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
    from collections import Counter
    from nltk.corpus import stopwords

    nltk.download('stopwords')

    # Load pre-trained Word2Vec model
    model_path = "GoogleNews-vectors-negative300.bin"  # Replace with your path
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Stop words
    stop_words = set(stopwords.words('english'))

    # Preprocess text
    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and word in model]
        return tokens

    # Process groups
    processed_groups = [list(map(preprocess_text, group)) for group in groups]

    # Cluster similar words to handle synonyms
    all_tokens = [word for group in processed_groups for doc in group for word in doc]
    unique_tokens = list(set(all_tokens))
    token_embeddings = np.array([model[word] for word in unique_tokens if word in model]).astype('float64')

    # Cluster tokens into groups
    num_clusters = num_topics * 5  # Use more clusters than topics
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(token_embeddings)

    # Map words to cluster labels
    token_to_cluster = {token: kmeans.predict([model[token].astype('float64')])[0] for token in unique_tokens if token in model}

    # Find the closest and diverse words to each cluster center
    cluster_centers = kmeans.cluster_centers_
    cluster_to_closest_words = {}

    for cluster_idx, center in enumerate(cluster_centers):
        cluster_words = [word for word in unique_tokens if token_to_cluster.get(word) == cluster_idx]
        cluster_word_distances = [(word, np.linalg.norm(model[word] - center)) for word in cluster_words if word in model]
        cluster_word_distances.sort(key=lambda x: x[1])  # Sort by distance to cluster center

        # Select the top 20 words, ensuring diversity
        selected_words = []
        for word, _ in cluster_word_distances:
            if len(selected_words) >= 20:
                break
            if all(np.linalg.norm(model[word] - model[w]) > 0.3 for w in selected_words):  # Ensure diversity
                selected_words.append(word)

        cluster_to_closest_words[cluster_idx] = selected_words

    # Replace tokens with cluster labels in documents
    def replace_with_clusters(tokens):
        return [str(token_to_cluster[word]) for word in tokens if word in token_to_cluster]

    clustered_groups = [[replace_with_clusters(doc) for doc in group] for group in processed_groups]

    # Create dictionary and corpus
    all_texts = [doc for group in clustered_groups for doc in group]
    dictionary = corpora.Dictionary(all_texts)
    corpus_groups = [[dictionary.doc2bow(doc) for doc in group] for group in clustered_groups]

    # Normalize corpus by group size
    group_sizes = [len(group) for group in corpus_groups]
    normalized_corpora = []
    for corpus, size in zip(corpus_groups, group_sizes):
        scaling_factor = 1 / size
        normalized_corpus = [
            [(term_id, count * scaling_factor) for term_id, count in doc] for doc in corpus
        ]
        normalized_corpora.append(normalized_corpus)

    # Combine normalized corpora
    combined_corpus = [doc for corpus in normalized_corpora for doc in corpus]

    # Train LDA model
    lda_model = LdaModel(corpus=combined_corpus, id2word=dictionary, num_topics=num_topics, passes=15)

    # Get topic distribution for each group
    group_topic_distributions = []
    for corpus in corpus_groups:
        group_topics = []
        for doc in corpus:
            topic_distribution = lda_model.get_document_topics(doc)
            top_topic = max(topic_distribution, key=lambda x: x[1])[0]  # Get the most probable topic
            group_topics.append(top_topic)
        group_topic_distributions.append(group_topics)

    # Cross-group topic analysis
    group_common_topics = []
    for group_topics in group_topic_distributions:
        group_common_topics.extend(group_topics)

    # Count topic frequencies across groups
    topic_counter = Counter(group_common_topics)
    most_common_topic = topic_counter.most_common(1)[0]  # Most common topic

    # Output results
    print("主题关键词:")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"主题 {idx}: {topic}")

    print("\n每个组的主题分布:")
    for group_idx, group_topics in enumerate(group_topic_distributions):
        print(f"组 {group_idx + 1}: {Counter(group_topics)}")

    print(f"\n跨组出现最多的主题是主题 {most_common_topic[0]}，出现次数为 {most_common_topic[1]} 次。")

    print("\n每个聚类的前20个单词（尽可能分散）:")
    for cluster_idx, closest_words in cluster_to_closest_words.items():
        print(f"聚类 {cluster_idx}: {', '.join(closest_words)}")



def extract_functions_from_file(filepath, prefix):
    """
    提取文件中的所有函数及其相关的 import 语句，并返回字典列表
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # 解析源代码
    tree = ast.parse(content)
    
    functions = []
    imports = []
    
    # 遍历所有节点
    for node in ast.walk(tree):
        # 如果是 import 语句
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            imports.append(ast.unparse(node))
        
        # 如果是函数定义
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            function_code = ast.unparse(node)
            function_with_imports = "\n".join(imports) + "\n" + function_code
            # 使用路径前缀生成新的键名
            prefixed_name = f"{prefix}-{function_name}"
            functions.append({prefixed_name: function_with_imports})
    
    return functions

def extract_functions_from_directory(directory):
    """
    遍历目录中的所有 Python 文件并提取所有函数
    """
    all_functions = []
    
    # 遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # 仅处理 Python 文件
                file_path = os.path.join(root, file)
                # 将路径作为 prefix 传递
                functions = extract_functions_from_file(file_path, prefix=directory)
                all_functions.extend(functions)
    
    return all_functions


def extract_imports_and_used_items_from_code(code):
    dependencies = {}  # 存储 {module: {imported_items}} 结构
    used_items = set()  # 存储代码实际使用的函数、类、变量名

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # 处理 import xxx
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    dependencies.setdefault(module_name, set())

            # 处理 from xxx import yyy
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module
                    if module_name not in dependencies:
                        dependencies[module_name] = set()
                    for alias in node.names:
                        dependencies[module_name].add(alias.name)

            # 记录代码中实际使用到的变量、函数、类
            elif isinstance(node, ast.Name):  # 变量/函数调用
                used_items.add(node.id)
            elif isinstance(node, ast.Attribute):  # obj.func() 这种调用
                used_items.add(node.attr)

    except Exception as e:
        print(f"Failed to parse code: {e}")
        return {}

    # 过滤未使用的导入项
    filtered_dependencies = {}
    for module, items in dependencies.items():
        used_imports = {item for item in items if item in used_items}  # 只保留被使用的项
        if used_imports or not items:  # 如果是 `import xxx` 形式也保留
            filtered_dependencies[module] = used_imports

    return filtered_dependencies

def parse_imports(tree):
    """解析import和from ... import语句，返回包名和模块路径映射"""
    import_map = {}
    alias_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_map[alias.name] = alias.name  # 直接import
                if alias.asname:
                    alias_map[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                import_map[alias.name] = full_name
                if alias.asname:
                    alias_map[alias.asname] = full_name
    return import_map, alias_map

def extract_functions(tree):
    """提取代码中的所有函数调用，包括XXX.YYY.ZZZ格式"""
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                functions.add(node.func.id)  # 直接函数调用
            elif isinstance(node.func, ast.Attribute):
                parts = []
                attr = node.func
                while isinstance(attr, ast.Attribute):
                    parts.append(attr.attr)
                    attr = attr.value
                if isinstance(attr, ast.Name):
                    parts.append(attr.id)
                functions.add(".".join(reversed(parts)))
    return functions

def match_imports(import_map, alias_map, functions):
    """匹配import的包和实际调用的函数，并格式化输出"""
    matched = {}
    for func in functions:
        parts = func.split(".")
        base = parts[0]  # 获取函数或属性的根
        actual_base = alias_map.get(base, base)  # 解析别名
        if actual_base in import_map:
            matched[func] = import_map[actual_base] + "." + ".".join(parts[1:]) if len(parts) > 1 else import_map[actual_base]
    return matched

def analyze_code(code):
    tree = ast.parse(code)
    import_map, alias_map = parse_imports(tree)
    functions = extract_functions(tree)
    return match_imports(import_map, alias_map, functions)

def extract_function_body(file_path, target_function):
    """从指定文件中提取目标函数的函数体"""
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    # 遍历所有函数定义节点
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == target_function:
                # 函数体内容
                function_body = ast.unparse(node)
                return function_body
    return None

def extract_class_or_function_body(file_path, target_name):
    """从指定文件中提取目标类或函数的内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    # 遍历所有节点
    for node in ast.walk(tree):
        # 检查是否为目标类
        if isinstance(node, ast.ClassDef) and node.name == target_name:
            class_body = ast.unparse(node)
            return class_body
        
        # 检查是否为目标函数
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            function_body = ast.unparse(node)
            return function_body
    return None

def find_target_class_or_function_in_directory(directory, target_name):
    """在目录中的所有Python文件中查找目标类或函数，并输出其内容"""
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                content = extract_class_or_function_body(file_path, target_name)
                if content:
                    res.append(content)
    return res

def find_file_in_folder(folder_path, target_file_name):
    """在文件夹中查找目标文件并返回其内容"""
    for root, dirs, files in os.walk(folder_path):
        if target_file_name in files:
            file_path = os.path.join(root, target_file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()  # 返回文件内容
    return ''  # 如果未找到文件，返回空字符串

def get_incontext(directory_path, target_name):
    incontext_content = find_file_in_folder(directory_path, target_name+'.py')
    if incontext_content == '':
        incontext_content = find_target_class_or_function_in_directory(directory_path, target_name)
    return incontext_content


def parse_patch(patch_content):
    function_dict = {}
    
    # 分割不同的补丁块（每个文件一个补丁）
    patches = re.split(r'(diff --git .*\n)', patch_content)[1:]
    
    for i in range(0, len(patches), 2):
        header = patches[i]
        content = patches[i+1] if i+1 < len(patches) else ''
        
        # 分割每个补丁块中的hunk
        hunks = re.split(r'(^@@ -\d+,\d+ \+\d+,\d+ @@.*\n)', content, flags=re.MULTILINE)
        
        current_function = None
        for j in range(1, len(hunks), 2):
            hunk_header = hunks[j].strip()
            hunk_body = hunks[j+1]
            
            # 从hunk头提取函数名
            func_match = re.search(r'@@.*@@\s*(.*?)(?:\s*\{)?\s*$', hunk_header)
            if func_match:
                raw_func = func_match.group(1)
                # 提取规范化的函数名（类名::方法名）
                normalized_func = re.match(r'^([\w:]+(?:\s*<.*>)?)\s*(?:\(|{)', raw_func)
                if normalized_func:
                    current_function = normalized_func.group(1)
                else:
                    current_function = raw_func.split('(')[0].strip()
            
            if not current_function:
                continue
            
            # 处理hunk中的代码行（修改前的内容）
            code_lines = []
            for line in hunk_body.split('\n'):
                # 捕获被删除的行（-）和未修改的上下文行（空格）
                if line.startswith(('-', ' ')):
                    # 去除符号并保留缩进
                    code_line = line[1:] if len(line) > 0 else ''
                    code_lines.append(code_line)
            
            if code_lines:
                if current_function not in function_dict:
                    function_dict[current_function] = []
                function_dict[current_function].extend(code_lines)
    
    # 合并为完整的函数代码字符串
    for func in function_dict:
        # 清理空行并保留原始缩进
        cleaned_lines = []
        for line in function_dict[func]:
            stripped = line.rstrip()  # 保留行首缩进，只去掉行尾空格
            if stripped:  # 过滤空行
                cleaned_lines.append(stripped)
        
        # 合并为单个字符串，保留换行和缩进结构
        function_dict[func] = '\n'.join(cleaned_lines)
    
    return function_dict
def process_patches_directory(directory_path):
    """
    处理整个目录下的所有patch文件
    :param directory_path: 存放.patch文件的目录路径
    :return: 包含所有解析结果的列表
    """
    result_list = []
    
    # 获取目录下所有文件
    for filename in os.listdir(directory_path):
        # 只处理.patch结尾的文件
        if filename.endswith('.patch'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    patch_content = f.read()
                
                # 解析patch文件
                parsed_result = parse_patch(patch_content)
                
                # 将结果添加到列表
                result_list.append({
                    "filename": filename,
                    "result": parsed_result
                })
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue
    
    return result_list
































