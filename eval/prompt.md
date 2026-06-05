## 生成问题的提示词 

请帮我生成一组用于评估当前RAG的问题集
要求：
1 参考eval\hak180产品安全手册.md
2  格式 csv
3  列头 question, ground_truth
4  生成个10对问题和答案  
5  保存在 eval目录下 名称为 qa.csv
6  编码格式要求： UTF-8 BOM 编码
7  把  "HAK 180烫金机:" 作为问题开头，明确问题的主体。


## 生成评估程序的提示词
请帮我生成一个基于ragas 的评估程序，用于评估当前项目的rag流程。

要求:
1   读取我的问题集 eval\qa.csv 
2   用问题集的question 去调用我的rag流程，流程入口 knowledge\processor\query_process\main_graph.py中的query_app
     获取最终的answer 和 context（ 从state中提取 reranked_docs )
3   调用ragas框架，使用对以下5个指标进行评估：Faithfulness，Answer Relevancy，Context Precision ， Context Recall ，Answer Correctness
4   把最终评估结果写入一个文件 eval\qa_eval.csv
    列头为   question,context,answer ,ground_truth, faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
5  csv 文件编码格式要求： UTF-8 BOM 
6  要求每个方法要有详细注释 
7  把每个步骤的核心函数用 step_1_xx \step_2_xx  .. 来命名
8  评估程序保存在eval目录下，名称为 eval.py
9  评估过程中需要的模型工具可以使用knowledge\utils 中的工具
10  只生成该评估程序，不要修改其他已有程序