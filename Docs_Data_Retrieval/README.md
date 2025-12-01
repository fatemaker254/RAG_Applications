## This project shows a RAG pipeline using langchain which is trained on the book "DISTRIBUTED OPERATING SYSTEMS" by Pradeep K. Sinha
![alt text](images/image.png)

## Parts in which the files are broken ar as follows:

# The markdownConverter.py file converts any pdf file uploaded to the folder to LLM ready markdown file for further processes like chunking and so on.

![alt text](images/image-1.png)

# The above image shows the embedding process which is scripted in the file create_database.py

# The most relevant embeddings are compared and found from the chromaDB. The implementation is shown in the compare_embeddings.py file.

The reference from the source data can be found in the metadata of each of the doc. 
# Steps to run 
1. install the dependencies(pip install -r requirements.txt)
3. Execute the create_database.py file 
4. Execute the query_data.py file to get the output 