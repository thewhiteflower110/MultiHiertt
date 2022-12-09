import json
import operator
import tqdm as tqdm
n=[10]

for ncount in n:
	# Opening JSON file
	f = open('FB_dev.json')
	  
	# returns JSON object as 
	# a dictionary
	data = json.load(f)
	  
	# Iterating through the json
	# list
	max_text_evidence=0
	max_table_evidence=0
	table_c=0
	text_c=0
	all_files_recall = 0.0
	all_files_f1 = 0.0
	all_files_Precision = 0.0
	c=0
	c1=0
	table_recall=0.0
	table_f1=0.0
	table_precision=0.0
	text_recall=0.0
	text_f1=0.0
	text_precision=0.0

	for i in data:
	    table_evidence=i['qa']['table_evidence']
	    text_evidence=i['qa']['text_evidence']

	    max_text_evidence=max(max_text_evidence,len(text_evidence))
	    max_table_evidence=max(max_table_evidence,len(table_evidence))

	    table_retrieved_all={}
	    text_retrieved_all={}

	    for j in i['table_retrieved_all']:
	    	table_retrieved_all[j['score']]	= j['ind']
	    
	    sorted(table_retrieved_all, key = table_retrieved_all.get,reverse = True)

	    for j in i['text_retrieved_all']:
	    	text_retrieved_all[j['score']]	= j['ind']
	    
	    sorted(text_retrieved_all, key = text_retrieved_all.get,reverse = True)

	    table_retrieved_all = dict(list(table_retrieved_all.items())[0: ncount])
	    text_retrieved_all = dict(list(text_retrieved_all.items())[0: ncount])

	    ans_table_retrieved_all=[]
	    ans_text_retrieved_all=[]
	    
	    for i in table_retrieved_all.values():
	    	ans_table_retrieved_all.append(i)

	    for i in text_retrieved_all.values():
	    	ans_text_retrieved_all.append(i)

	    #import pdb; pdb.set_trace();
	    correct_table_ids = len(set(ans_table_retrieved_all).intersection(set(table_evidence))) 
	    fp_table = len(ans_table_retrieved_all)-correct_table_ids

	    correct_text_ids = len(set(ans_text_retrieved_all).intersection(set(text_evidence))) 
	    fp_text = len(ans_text_retrieved_all)-correct_text_ids

	    if(len(table_evidence)>0):
	        table_recall +=(correct_table_ids) / (len(table_evidence) )
	        table_c+=1
	    else:
	        c1+=1

	    if(len(text_evidence)>0):
	        text_recall +=(correct_text_ids) / len(text_evidence) 
	        text_c+=1
	    else:
	        c+=1

	    #all_files_recall += (correct_text_ids) / (len(text_evidence))
	    all_files_recall += (correct_table_ids+correct_text_ids) / (len(table_evidence) + len(text_evidence))
	    #break
		
	    Precision_table = correct_table_ids/(correct_table_ids+fp_table)
	    Precision_text = correct_text_ids/(correct_text_ids+fp_text)
	    table_precision+=Precision_table
	    text_precision+=Precision_text
	    
	    #all_files_Precision+=(Precision_text)
	    all_files_Precision+=(Precision_text+Precision_table)


	resultant_recall = text_recall / len(data)
	resultant_table_recall = table_recall / (len(data)-c1)
	resultant_text_recall = text_recall / (len(data)-c)
	
	resultant_precision = all_files_Precision / len(data)
	resultant_table_precision = table_precision / len(data)
	resultant_text_precision = text_precision / len(data)


	resultant_f1 = 2*(resultant_recall * resultant_precision) / (resultant_recall + resultant_precision)
	resultant_table_f1 = 2*(resultant_table_recall * resultant_table_precision) / (resultant_table_recall + resultant_table_precision)
	resultant_text_f1 = 2*(resultant_text_recall * resultant_text_precision) / (resultant_text_recall + resultant_text_precision)

	print("ncount:",ncount)
	print("Recall: ",resultant_recall)
	print("Precision: ",resultant_precision)
	print("F1: ",resultant_f1)
	print("-"*20)
	print("resultant_table_recall:",resultant_table_recall)
	print("resultant_text_recall:",resultant_text_recall)
	print("resultant_table_precision:",resultant_table_precision)
	print("resultant_text_precision:",resultant_text_precision)

	print("resultant_table_f1:",resultant_table_f1)
	print("resultant_text_f1:",resultant_text_f1)
	print("Table:",table_c,c1,table_c-c1)
	print("Text:",text_c,c,text_c-c)
	
	# Closing file
	f.close()