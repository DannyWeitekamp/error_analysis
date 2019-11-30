import os,sys
import pandas
import copy
from pprint import pprint
import numpy as np
import re
# import matplotlib.pyplot as plt
# from matplotlib import colors as mcolors
import xml.etree.cElementTree as et 
import argparse
import json


# All possible error types made by the students:
ERROR_TYPES = {
    "1100": "correct", "1101": "correct",
    "1000": "incorrect", 
    "1001": "misapplied",
    "0000": "out of graph", "0001": "out of graph", "0100": "out of graph", "0101": "out of graph", 
    "1110": "correct repeat", 
    "1010": "repeat", "1011": "repeat", "1111": "repeat", 
    "0110": "where error", "0111": "where error",
    "0011": "when error", 
    "0010": "wild error"
}


''' Read in a transaction csv file '''
def transaction_file_to_df(path):
    df = pandas.read_csv(path, sep='\t', lineterminator='\n', skip_blank_lines=True).replace({r'\r': ''}, regex=True)
    df = df.rename(index=int, columns={a:a.rstrip() for a in df.keys()})
    return df


'''
extract_from_brd: returns a dataframe with information of all edges (correct steps) in brd file
Input: directory path to brd files
Output: a dataframe with columns of SAI, ID, source and destination of correct edges in brd
'''
def extract_from_brd(path):

    parsedXML = et.parse(path)
    dfcols = ["match_s","match_a","match_i","match_actor","match_ID","match_source","match_dest"]
    df_xml = pandas.DataFrame(columns= dfcols)

    for e in parsedXML.findall(".//edge"):    
        ID = e.find('./actionLabel/uniqueID').text
        source = e.find('./sourceID').text
        dest = e.find('./destID').text

        s = e.find('.actionLabel/matchers/Selection/matcher/matcherParameter').text
        a = e.find('.actionLabel/matchers/Action/matcher/matcherParameter').text
        i =  e.find('.actionLabel/matchers/Input/matcher/matcherParameter').text  
        actor = e.find('.actionLabel/matchers/Actor').text 
 
        df_xml = df_xml.append(
                pandas.Series([s, a, i, actor, ID, source, dest], index= dfcols),
                ignore_index=True)
        
        df_xml[['match_ID', 'match_source', 'match_dest']] = df_xml[['match_ID', 'match_source', 'match_dest']].apply(pandas.to_numeric)
    
    return(df_xml)
    



'''
update_input, update_action, update_selection: clean the SAI of edges in brds so they match the SAI of students in log
Input: one row in the brd dataframe before cleaning
Output: one row in the brd dataframe with cleaned SAI
'''

# clean the edge's input in humans brds
def update_input(row): 
    if(row['match_i'] == 'x'):
     #   return("I need to convert these fractions before solving.: true")
        return('x')
    
    if(row['match_i'] == 'v'):
        return("I need to convert these fractions before solving.: false")
    
    else:        
        return(str(row['match_i']))
    
    
# clean the edge's action in humans brd        
def update_action(row): 
    if(row['match_a'] == 'UpdateTextArea'):
        return('UpdateTextArea') 
    
    if(row['match_a'] == 'ButtonPressed'): 
           return('ButtonPressed')
        
        
# clean the edge's selection in humans brd         
def update_selection(row): 
    if(row['match_s'] == 'done'): 
        return('done') 
    elif(row['match_s'][10] == '8'): 
        return('check_convert') 
    else: 
        if(row['match_s'][10] == '4'): 
            box = '3' 
        if(row['match_s'][10] == '5'): 
            box = '4' 
        if(row['match_s'][10] == '6'): 
            box = '5' 
        if(row['match_s'][13] == '0'): 
            pos = 'num' 
        if(row['match_s'][13] == '1'): 
            pos = 'den' 
        return(pos+box)
           


''' 
clean_extract: clean variable names in CTAT interface so they match the ones used by human students
Input: original brd dataframe from extract_from_brd
Output: cleaned brd dataframe 
'''
def clean_extract(extract,rename_map={}):
    new = extract.copy()
    s_m = rename_map.get('selection',{})
    a_m = rename_map.get('action',{})
    i_m = rename_map.get('input',{}) 
    new.insert(0, 'match_s_new', new["match_s"].apply(lambda x : s_m[x] if x in s_m else x))
    new.insert(0, 'match_a_new', new["match_a"].apply(lambda x : a_m[x] if x in a_m else x))
    new.insert(0, 'match_i_new', new["match_i"].apply(lambda x : i_m[x] if x in i_m else x))
    # new['match_s_new'] = new.apply(update_selection, axis=1)
    # new['match_a_new'] = new.apply(update_action, axis=1)
    # new['match_i_new'] = new.apply(update_input, axis=1)
    return(new)



'''
get_correct_transaction: return only student-performed SAIs that are correct
Input: original transactions of all problems from one student
Output: correct transactions of all problems from one student
'''
def get_correct_transaction(transaction):
    return (transaction[(transaction.Outcome == 'CORRECT') 
                & (transaction.Action != 'setVisible')
                & (transaction.Input != '+')])



''' 
check_transactions: filter out invalid transactions. If transactions end incorrectly, return False 
Input: 
    -df: student slice, a dataframe of a series of transactions that end with correct steps
    -graph: graph generated from the brd dataframe
Output: true if valid
'''

def check_transactions(df, graph):
    # transactions that end incorrectly
    
    if( (df['Outcome'].iloc[-1] == 'INCORRECT') | (df['Outcome'].iloc[-1] == 'HINT')): return(False)
    return(True)
    
    
'''
clean_transaction: remove correct steps in transactions that are not in the brds
Input: original student slice 
Outpud: student slice without correct steps that are not in brds 
'''
def clean_transaction(df):
    r = -1
    for _, table_row in df.iterrows():
        
        if(table_row['Input'] == 'I need to convert these fractions before solving.: false' and table_row['Outcome'] == 'CORRECT'):
            r = _
    if r!= -1:
        df = df.drop([r])
            
    return(df)

'''
get_graph_from_extract: return a graph created from brd dataframe
Input: brd dataframe 
Output: graph as a pair of mappings. Keys are edges in brd, values are its downstream neighbouring edges. 
'''
def get_graph_from_extract(extract):
    graph = {}
    tl = set()
    for _,table_row in extract.iterrows():
        ID = table_row['match_ID']
        source = table_row['match_source']
        dest = table_row['match_dest']
        t = (ID, source, dest)
        tl.add(t)
        graph[t] = []
    for i in graph:
        d = i[2]
        for j in tl:
            if (j[1] == d):
                graph[i].append(j)
    return(graph)


def get_first_correct_IDs(row, extract):
    lst = [] 
    S, A, I = row['Selection'], row['Action'], row['Input']
    
    for _,table_row in extract.iterrows():
        
        if(S == table_row['match_s_new'] and A == table_row['match_a_new'] and I == table_row['match_i_new']):
            ID, source, dest = table_row['match_ID'], table_row['match_source'], table_row['match_dest']
            lst.append((ID, source, dest))
    return(lst)



'''
get_node_SAIs: return a dictionary of SAI for each unique node 
Input: 
    -graph: graph generated from the brd file using get_graph
    -cleaned_extract: cleaned brd dataframe from clean_extract
Output: a dictionary of SAI for each node in graph 
'''
def get_node_SAIs(graph, cleaned_extract):
    correct_sai = {}
    for (ID, s, d) in graph:
        for _,table_row in cleaned_extract.iterrows():
            if(ID == table_row['match_ID'] and s == table_row['match_source'] and d == table_row['match_dest']):
                correct_sai[(ID, s, d)] = ((table_row['match_s_new'], table_row['match_a_new'], table_row['match_i_new']))
    return(correct_sai)



''' Find all starting nodes in a graph ''' 
def find_beginnings(graph):
    s = set()
    for i in graph:
        start = True
        for j in graph:
            if i in graph[j]:
                start = False
        if start:
            s.add(i)           
    return(s)
   
    
    
'''
find_current_index: for student's incorrect steps, identify the next correct edge in brd that the student 
is working toward      
Input: 
    -correct: list of indices for correct steps for a student slice
    -incorrect: list of indices for incorrect steps for a student slice
Output: a dictionary with keys being incorrect indices and values being the indices for next correct step 
for that incorrect index.

'''    
    
def find_current_index(correct, incorrect):
    res = {}
    correct_copy = copy.deepcopy(correct)
    
    
    # use an incorrect copy
    incorrect_copy = copy.deepcopy(incorrect)
    while (incorrect_copy and (incorrect_copy[-1] > correct_copy[-1])):
        res[incorrect_copy[-1]] = correct_copy[-1]
        incorrect_copy.pop()
        

    
    for i in range(len(incorrect_copy)):
        if (incorrect_copy[i] < correct_copy[0]):
            res[incorrect_copy[i]] = correct_copy[0]
        else:
            while(correct_copy and (incorrect_copy[i] > correct_copy[0])):
                correct_copy.pop(0)
            res[incorrect_copy[i]] = correct_copy[0]
    return(res)



'''
find_last_index: for student's incorrect steps, identify the immediate last edge in brd 
that the student has done correct 
Input: 
    -correct: list of indices for correct steps for a student slice
    -incorrect: list of indices for incorrect steps for a student slice
Output: a dictionary with keys being incorrect indices and values being the indices for last step in brd 
that the student has done correct 

'''
def find_last_index(correct, incorrect):
    incorrect_copy = copy.deepcopy(incorrect)
    while (incorrect_copy and (incorrect_copy[0] < correct[0])):
        incorrect_copy.pop(0)

    res = {}
    inc, cor = incorrect_copy[::-1], correct[::-1]
    for i in inc:
        if(i < cor[0]):
            while(i < cor[0]):
                cor.pop(0)
            res[i] = cor[0]
        else:
            res[i] = cor[0]
    return(res)



'''
search: return all downstream edges from a certain starting edge in brd graph using bfs
Input:
    -graph: graph generated from the brd dataframe
    -start: an edge to start searching downstream with
Output: a list of edges that is downstream from the start edge in the order of bfs 

'''
def search(graph, start):
    if start not in graph: return None
    visited, queue = [], [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(set(graph[node]) - set(visited))           
    return visited



'''
check_downstream: return true if an edge is in downstream 
Input:
    -graph: graph generated from the brd dataframe
    -up: the node to compare downstream with 
    -down: the node to check 
Output: return true if down is in downstream of up 
'''
def check_downstream(graph, up, down):
    if down in search(graph, up): return True
    return False 



'''
first_match: for student's correct step, return first edge in brd graph that matches it
Input:
    -df: a dataframe of a single row of one correct step made by student
    -lst: a list returned from search that contains all the downstream edges below a certain edge; 
    candidates to match with a certain correct student step 
    -d: dictionary returned from get_node_SAIs
    -last_node: the edge in brd that corresponds to last matched correct step made by student 
    -gaph: graph generated from the brd dataframe
Output: return first edge in brd graph that matches a correct step by student. If could not find one, return None

'''
def first_match(df, lst, d, last_node, graph):
    s_c, a_c, i_c = df['Selection'], df['Action'], df['Input']
    
    for node in lst:
        if check_downstream(graph, last_node, node):
            (s_l, a_l, i_l) = d[node] 
            # print(node,(s_c, a_c, i_c),(s_l, a_l, i_l))
            if (s_c, a_c, i_c) ==  (s_l, a_l, i_l):
                return node          
    return None       

'''
match_steps: match one students' SAI with cleaned brd file, update the new four columns
with binary values in dataframe 
Input: 
    -one_student: all step slices of a student including all problems in correct time order
    -cleaned_extract: cleaned brd from clean_extract
    -graph: graph generated from the brd dataframe
Output: a copy of student slice dataframe with new columns of binary values 

'''

def match_steps(one_student, cleaned_extract, graph):
    match = one_student.copy()
    match['SAI'], match['node'], match['downstream'], match['d_nodes'] = None, None, None, None
    match['S_current'], match['I_current'], match['S_downstream'], match['I_downstream'] = None, None, None, None
    
    
    #Add new columns error types and KC to df
    match['KC_toward'], match['error_type'] = None, None
    
    
    correct = get_correct_transaction(match) 
    correct_queue = []
    all_first_nodes = find_beginnings(graph)
    sai_dict = get_node_SAIs(graph, cleaned_extract)
    
    for _,table_row in correct.iterrows():
        correct_queue.append([_, table_row])

        
    # Match correct steps:    
    # for loop because might start with multiple nodes in brd
    for begin in all_first_nodes:
        
        SAI_d = {}
        done = set()
        
        last_node = begin
        correct_order = search(graph, begin)
        correct_transaction = {}
        correct_queue_copy = copy.deepcopy(correct_queue)
        can_match = True
        
        transaction_sel_set = set()
        while correct_queue_copy and correct_order:

            n = correct_queue_copy.pop(0)
            next_index, next_correct = n[0], n[1]
            
            s, a, i = next_correct['Selection'], next_correct['Action'], next_correct['Input']
            
            # print(s,a,i)
            if (s,a,i) in done:
            #deal with duplications
            
                first_index = SAI_d[(s,a,i)]
                correct_transaction[next_index] = correct_transaction[first_index]
                
                
            
            else:
                done.add((s,a,i))
                transaction_sel_set.add(s)
                SAI_d[(s,a,i)]= next_index
                
                f_match = first_match(next_correct, correct_order, sai_dict, last_node, graph)

            
                if f_match == None:
                    can_match = False
                    break
            
                
                while correct_order[0] != f_match:
                    correct_order.pop(0)
                correct_order.pop(0)
                    
                correct_transaction[next_index] = f_match
                last_node = f_match
        

        if can_match == True: break
            
    
    if can_match == False:  
        a = transaction_sel_set
        b = set([x[0] for x in sai_dict.values()])
        if(not a.issubset(b)):
            print("Cannot match step slice with selections %s to brd with selections %s" % (a,b))
            print("Consider using the --rename option")
        else:
            print("Can not match for this step_slice", match)
        return match
    
    
    else:
        
        index = sorted(list(correct_transaction.keys()))
       
        for i in index:
            match.at[i,'SAI'] = 'correct'
            match.at[i, 'node'] = correct_transaction[i]        
            match.at[i, 'S_current'], match.at[i, 'I_current'] = 1,1
       
        incorrect = list((match[match['Outcome'] == 'INCORRECT']).index)

        # match incorrect steps 
        if incorrect: 
    
            cur_index = find_current_index(index, incorrect)
            last_index = find_last_index(index, incorrect)


            for i in incorrect:
                # print("INCORRECT", i)
            
                match.at[i ,'SAI'] = 'incorrect'
                
                #KC = match.at[cur_index[i], 'KC (Field)'] #KC of the node working towards 
                #match.at[i, 'KC_toward'] = KC


                # Find current steps for incorrect transactions 
                if(i not in last_index.keys()):
                    match.at[i, 'node'] = list(find_beginnings(graph))
                
                
                if(i in last_index.keys()):
                    # print("last_index",last_index,i,last_index[i])
                    n_l = match.at[last_index[i], 'node'] 
                    match.at[i, 'node'] = graph[n_l]
                

                
                
                sel, inp = match.at[i, 'Selection'], match.at[i, 'Input']
                s_found_c, i_found_c = 0, 0
                
                match_s_c = None 
                for c in match.at[i, 'node']:
                    # print(c)
                    (s_c, a_c, i_c) = sai_dict[c]
                    # print(sai_dict)
                    # print(sai_dict[c])
                    if s_c == sel: 
                        s_found_c = 1
                        match_s_c = c
                        
                    if i_c == inp: i_found_c = 1
                match.at[i, 'S_current'], match.at[i, 'I_current'] = s_found_c, i_found_c
                        
                        
                downstream = set()
                # downstream for a particular branch
                if match_s_c != None:
                    down = graph[match_s_c]
                    for d in down: 
                        downstream.add(d)
                        
                else:
                
                    for node in match.at[i, 'node']:
                        down = graph[node]
                        for d in down: 
                            downstream.add(d)

                match.at[i, 'downstream'] = list(downstream)
                
                
                


                d_nodes = set()
                for n in match.at[i, 'downstream']:
                    d_nodes = d_nodes.union(search(graph, n))
     
                match.at[i, 'd_nodes'] = d_nodes
                s_found_d, i_found_d = 0, 0
                for dn in d_nodes:
                    (s_d, a_d, i_d) = sai_dict[dn]
                    if(sel == s_d):
                        s_found_d = 1    
                    if(inp == i_d):
                        i_found_d = 1

   
                match.at[i, 'S_downstream'], match.at[i, 'I_downstream']  = s_found_d, i_found_d
            

                e_type = str(s_found_c) + str(i_found_c) + str(s_found_d) + str(i_found_d) 
                match.at[i, 'error_type'] = ERROR_TYPES[e_type]
                # print("INCORRECT END", i, s_found_c, i_found_c, s_found_d, i_found_d)
            
        return match



'''
original_df: For cases that end in incorrect or hint, return the original df with new blank columns 
Input: all step slices for one student 
Output: all step slices for one student with new blank columns
'''
def original_df(one_student):

    match = one_student.copy()
    match['SAI'], match['node'], match['downstream'], match['d_nodes'] = None, None, None, None
    match['S_current'], match['I_current'], match['S_downstream'], match['I_downstream'] = None, None, None, None
    
     #Add error types and KC
    match['KC_toward'], match['error_type'] = None, None
    return match
    

'''
one_student_all_problems: match all problems from brd files for one student 
Input:
    -df: a dataframe of all problems for one student
    -directory: a directory path in which to find all brd files
Output: a new dataframe with new columns of binary values for this student 

'''
def one_student_all_problems(df, directory, stu,rename_map={}):
    
    new = pandas.DataFrame(columns = list(df.columns))
    new['SAI'], new['node'], new['downstream'], new['d_nodes'] = None, None, None, None
    new['S_current'], new['I_current'], new['S_downstream'], new['I_downstream'] = None, None, None, None   
    new['KC_toward'], new['error_type'] = None, None

    
    
    for problem in df['Problem Name'].unique(): # ? preserve order

        stu_slice = ( df[df['Problem Name'] == problem] ).copy()
        stu_slice = clean_transaction(stu_slice)

        if (problem != "InstructionSlide"):
            brd = extract_from_brd(directory + "/" + problem + ".brd")
            graph = get_graph_from_extract(brd)
            # print(graph)

            if check_transactions(stu_slice, graph):
                # print("GOOOOOOOOOOOOOOOOOOOOOOOOOOOD")
                tutor_SAI = clean_extract(brd,rename_map)
                # print("tutor_SAI",tutor_SAI[['match_s','match_s_new']])
                stu_match = match_steps(stu_slice, tutor_SAI, graph)
                #raise ValueError()
                new = new.append(stu_match)
            else:
                # print("BAAAAAAAAAAAAAAAAAAAAAAAAAAAD")
                orig_match = original_df(stu_slice)
                new = new.append(orig_match)

    return(new)

def print_load_bar(i,L):
    ticks = int(50*i/float(L))
    bar = "[" + "*" * ticks + " " * (50-ticks) + "]"
    sys.stdout.write("\r" + ("%i/%i" %(i,L)) + bar)
    sys.stdout.flush()
    
    
'''
generate_split_errors: take in paths to iso transactions and brd files to generate 
new dataframe with four new columns with binary values
Input:
    -transactions: a path for the iso transaction log
    -brd_path: a directory path in which to find the brd files whose names match the problem 
    names in iso transaction log
    -save_path: a path to save the new file generated
Output: a new dataframe of student transaction log with new columns of binary values 
    
'''
def generate_split_errors(transactions, brd_path, save_path, rename_map={},verbosity=1, requirements=[]):
    t = transaction_file_to_df(transactions)
    for column,value in requirements:
        t =  (t[t[column] == value]).reset_index(drop=True)

    
    df = pandas.DataFrame(columns = list(t.columns))
    df['SAI'], df['node'], df['downstream'], df['d_nodes'] = None, None, None, None
    df['S_current'], df['I_current'], df['S_downstream'], df['I_downstream'] = None, None, None, None
    
    df['KC_toward'], df['error_type'] = None, None
    
    students = t['Anon Student Id'].unique()
    for i,stu in enumerate(students):
        if(verbosity >= 1): print_load_bar(i,len(students))
            
        p = t[t['Anon Student Id'] == stu]
    
        new = one_student_all_problems(p, brd_path, stu, rename_map=rename_map)
        df = df.append(new)
        with open(save_path, 'a') as f:
             new.to_csv(f, header=False,sep='\t')

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotates a transaction file by split error types.')
    parser.add_argument('transactions', type=str, metavar="<student_transactions>.txt",
        help="A tab delimited table with the student transactions (standard CTAT logging output).")
    parser.add_argument('brds', type=str, metavar="<behavior_graph>.brd",
        help="The behavior graph for the example tracing tutor that the data was recorded from.")
    parser.add_argument('output', type=str, metavar="<output>",
        help="The output tab delimted table annotated by different error types.")
    parser.add_argument('-r','--rename' , default=None, dest = "rename_map", 
        help='A JSON file with dictionary {"selection" : <mapping>, "action" : <mapping>, "input" : <mapping>} \
            where each <mapping> is a dictionary that maps value in the brd to their corresponding value in \
            the transaction file. Useful if brds and transactions use different naming conventions.')
    parser.add_argument("--require", default=None, dest='requirements',nargs='+',
        help="A set of requirements <column_name1>=<value1> ... <column_nameN>=<valueN> which must be satisfied for a row to be included")
    parser.add_argument('-v','--verbosity' , default=1, dest = "verbosity",
        help='0 or 1')

    try:
        args = parser.parse_args(sys.argv[1:])
        # args.setattr(args, "training", args.training[0]) # dunno why it comes in a list
        
    except Exception as e:
        print(e)
        parser.print_usage()
        sys.exit()

    rename_map = {}
    if args.rename_map != None:
        with open(args.rename_map) as json_file:
            rename_map = json.load(json_file)
    

    requirements = [re.split("=+",x)[:2] for x in args.requirements]
    # print(requirements,args.requirements)

    
    generate_split_errors(transactions=args.transactions,
                          brd_path=args.brds,
                          save_path=args.output,
                          rename_map=rename_map,
                          verbosity=args.verbosity,
                          requirements=requirements)


    
    


