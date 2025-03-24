from sparql.Utils import convertToTurtle, list_to_string_triples
from sparql.Filter_Triples import Filter_Triples
from nlp.parsers import *
from openai import OpenAI
from dotenv import load_dotenv
import warnings
import re
from context.ContextLLM_ours import *
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from configs import NUMBER_HOPS,LIMIT_BY_PROPERTY,FILTER_GRAPH,RELEVANCE_THRESHOLD,MAX_HITS_RATE,PRINT_HITS,TEMPERATURE_TRANSLATE,TEMPERATURE_SELECT,TEMPERATURE_FINAL_ANSWER,USE_A_BOX_INDEX
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import json
import warnings
import re
from pydantic import BaseModel
from typing import Dict
import openai # CBORG API Proxy Server is OpenAI-compatible through the openai module
import os
#OpenAI
load_dotenv()

client = openai.OpenAI(
    api_key=os.environ.get('CBORG_API_KEY'), # Please do not store your API key in the code
    base_url="https://api.cborg.lbl.gov" # Local clients can also use https://api-local.cborg.lbl.gov
)
#client = openai.OpenAI(
#    api_key=os.environ.get('OPENAI_API_KEY'), # Please do not store your API key in the code
#    base_url="https://api.llama-api.com" # Local clients can also use https://api-local.cborg.lbl.gov
#)

import csv
import os
from datetime import datetime





class QuestionHandler:

    def __init__(self, endpoint, endpoint_t_box, t_box_index, normalizer, messagesTranslater=ContextTranslator(""),
                 messagesNL=ContextNLGenerator(), generalConversation=ContextDialog(),
                 messagesChooseBest=ContextChooseBestSPARQL(""), a_box_index=None, model_name="gpt-3.5-turbo-16k") -> None:
        self.endpoint = endpoint
        self.endpoint_t_box = endpoint_t_box
        self.t_box_index = t_box_index
        self.normalizer = normalizer
        self.messagesTranslater = messagesTranslater
        self.messagesNL = messagesNL
        self.generalConversation = generalConversation
        self.messagesChooseBest = messagesChooseBest
        self.a_box_index = a_box_index
        self.model_name = model_name
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.token_log_file = "token_usage_log_ours.csv"  # Define CSV file path
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_token_usage(self, model_name, question, final_query, num_rows):
        """Logs model name, question, total prompt tokens, final SPARQL query, and number of rows in the result."""
        file_exists = os.path.isfile(self.token_log_file)
        with open(self.token_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Model", "Question", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Final SPARQL Query", "Number of Rows"])
            writer.writerow([datetime.now().isoformat(), model_name, question, self.prompt_tokens, self.completion_tokens, self.total_tokens, final_query, num_rows])

    def call_llm(self, messages, temperature, n):
        print("messages", messages)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        self.prompt_tokens += response.usage.prompt_tokens  # Capture token usage
        self.completion_tokens += response.usage.completion_tokens  # Capture token usage
        self.total_tokens += response.usage.total_tokens

        print("Model used", self.model_name, "response:", response.choices[-1].message.content)
        return response # Return tokens along with response

    def textToSPARQL(self, question, ttl, rag=None):
        self.messagesTranslater.changeGraph(ttl, rag=rag)
        self.messagesChooseBest.changeGraph(ttl)

        self.messagesTranslater.add({"role": "user", "content": question})
        self.generalConversation.add({"role": "user", "content": question})

        # Call LLM to generate multiple SPARQL queries
        completion = self.call_llm(self.messagesTranslater.to_list(), TEMPERATURE_TRANSLATE, 5)

        results = []
        sparqls = []
        structured_results = []

        try:
            response_content = completion.choices[-1].message.content.strip()

            # Remove any leading "json" keyword if present
            if response_content.lower().startswith("json"):
                response_content = response_content[4:].strip()  # Remove "json" and trim whitespace

            # Ensure the response contains valid JSON format
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                response_content = match.group(0)  # Extract the JSON object

            queries_json = json.loads(response_content)  # Parse response as JSON

        except json.JSONDecodeError:
            print("Error: LLM response is not a valid JSON.")
            print("Trying again")
            completion= self.call_llm(self.messagesTranslater.to_list(), TEMPERATURE_TRANSLATE, 5)
            
            response_content = completion.choices[-1].message.content.strip()

            if response_content.lower().startswith("json"):
                response_content = response_content[4:].strip()

            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                response_content = match.group(0)

            queries_json = json.loads(response_content)

        # Iterate over the 5 generated queries
        for idx, query in queries_json.items():
            sparql = query.strip()  # Remove extra spaces/newlines
            
            # Ensure query is correctly formatted
            if not sparql.lower().startswith(("prefix", "select", "ask", "construct", "describe")):
                print(f"Skipping malformed SPARQL query at index {idx}: {sparql}")
                continue
            
            print(f"Executing SPARQL query {idx}:\n{sparql}\n")

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    question_formulated, query_results = self.endpoint.struct_result_query(sparql)
                    if query_results is not None:
                        structured_results.append(question_formulated)
                        results.append(query_results)
                        sparqls.append(sparql)
                except Exception as e:
                    print(f"Error executing query {idx}: {e}")
                    continue

        # If no valid SPARQL queries, return None
        if not results:
            self.messagesTranslater.add({"role": "assistant", "content": ""})
            return None

        # Let LLM choose the best query
        self.messagesChooseBest.changeQuestion(question, structured_results)
        self.messagesChooseBest.add({"role": "user", "content": question})
        completion = self.call_llm(self.messagesChooseBest.to_list(), TEMPERATURE_SELECT, 1)
        selection = completion.choices[0].message.content

        try:
            selection_number = [int(s) for s in re.findall(r'\b\d+\b', selection)][0] - 1  # Convert to zero-based index
            if selection_number < 0 or selection_number >= len(sparqls):
                raise ValueError("Invalid selection index")
            sparql_selected = sparqls[selection_number]
            result_selected = results[selection_number]
        except Exception as e:
            print(f"Error in selection: {selection} - {e}")
            selection_number = 0
            sparql_selected = sparqls[0]
            result_selected = results[0]

        self.messagesTranslater.add({"role": "assistant",
                                    "content": f"""
                                        ```sparql
                                        {sparql_selected}
                                        ```
                                        """})

        # Compute the number of rows in the final query result
        num_rows = len(result_selected) if isinstance(result_selected, list) else 0

        # Log token usage along with the final SPARQL query and number of rows
        self.log_token_usage(self.model_name, question, sparql_selected, num_rows)

        return [sparqls, results, selection_number]


    def getRelatedTriples(self,question,index,number_hops=NUMBER_HOPS,limit_by_property=LIMIT_BY_PROPERTY):
        # print("getRelatedTriples: "+question)
        matchs = parseText(question,index,self.normalizer,self.endpoint)
        triples = ""
        nodes = []
        properties = []
        for match in matchs:
            # print("match: "+str(match))
            if match['content']['?type'] == 'property':
                properties.append(match['content']['?term'])
            else:
                nodes.append(match['content']['?term'])
            triples+= self.endpoint.describe(match["content"]["?term"],number_hops,limit_by_property)
        return triples,nodes,properties

    def getRelevantGraph(self,question,number_hops=NUMBER_HOPS,limit_by_property=LIMIT_BY_PROPERTY,filter_graph= FILTER_GRAPH,last_question=None,use_a_box_index=USE_A_BOX_INDEX):
        self.endpoint.visited_nodes = set()
        needed_nodes = []
        needed_properties = []
        hist_questions = question
        if last_question != None:
            hist_questions = last_question+"\n "+question
        triples,needed_nodes,needed_properties = self.getRelatedTriples(hist_questions,self.t_box_index)
        if use_a_box_index and self.a_box_index != None:
            # print("a_box_index")
            triples2,nodes,properties= self.getRelatedTriples(hist_questions,self.a_box_index,number_hops,limit_by_property)
            needed_nodes += nodes
            triples+= triples2
        # print("needed_nodes:"+str(needed_nodes))
        # print("needed_properties:"+str(needed_properties))
        triples+= self.endpoint_t_box.get_all_triples() #Put all relevant T-Box's triples in the selected triples set
        if filter_graph and len(triples) > 0:
            # print(triples)#aqui
            filter_triples = Filter_Triples(triples,self.embedding_function,relevance_threshold = RELEVANCE_THRESHOLD, max_hits_rate=MAX_HITS_RATE)
            selected_triples = filter_triples.filter_triples_relevance(question, print_hits=PRINT_HITS,needed_nodes=needed_nodes,needed_properties=needed_properties)
            triples = list_to_string_triples(selected_triples)
        # print("triples:\n"+triples)
        ttl = convertToTurtle(triples)
        self.endpoint.visited_nodes = set()
        return ttl


        

    def generateNLResponse(self,question,sparql_selected,results_selected):
        question_forumlated = f"""
        User question: "{question}";
        SPARQL query:
        ```sparql
        {sparql_selected}
        ```;
        JSON result set:
        ```json
        {results_selected}
        ```;
        """
        self.messagesNL.add({"role":"user","content":question_forumlated})
        #completion = self.client.chat.completions.create(model=self.model_name,messages=self.messagesNL.to_list(),temperature=TEMPERATURE_FINAL_ANSWER)
        completion=self.call_llm(self.messagesNL.to_list(),TEMPERATURE_FINAL_ANSWER,1)
        answer = completion.choices[0].message.content
        self.messagesNL.add({"role":"assistant","content":answer})
        if self.generalConversation[-1]["role"] == "assistant":
            self.generalConversation[-1]["content"] = self.generalConversation[-1]["content"] + f"\nSPARQL:```sparql\n{sparql_selected}```\n{answer}"
        else:
            self.generalConversation.add({"role":"assistant","content":f"\nSPARQL:```sparql\n{sparql_selected}```\n{answer}"})
        return answer
    

    def processQuestion(self,question,number_hops=NUMBER_HOPS,limit_by_property=LIMIT_BY_PROPERTY,filter_graph= FILTER_GRAPH,last_question=None):
        ttl = self.getRelevantGraph(question,number_hops,limit_by_property,filter_graph,last_question=last_question)
        textToSPARQL_return = self.textToSPARQL(question,ttl)
        if textToSPARQL_return != None:
            sparqls,results,selection_number = textToSPARQL_return
            sparql_selected = sparqls[selection_number]
            results_selected = results[selection_number]
            answer = self.generateNLResponse(question,sparql_selected,results_selected)
            llmAnswer = {'answer':answer,
                         'question':question,
                         'sparql':sparql_selected,
                         'fragments':ttl,
                         'sparqls':sparqls}
            return llmAnswer
        else:
            print("Não gerou consulta SPARQL válida")
            self.generalConversation.add({"role":"user","content":question})
            #completion = self.client.chat.completions.create(model=self.model_name, messages=self.generalConversation.to_list())
            completion=self.call_llm(self.generalConversation.to_list(),TEMPERATURE_FINAL_ANSWER,1)
            answer = completion.choices[0].message.content
            self.generalConversation.add({"role":"assistant","content":answer})
            return {'answer':answer,
                    'question':question,
                    'sparql':None,
                    'fragments':ttl,
                    'sparqls':None}

    @staticmethod
    def extractSPARQL(text):
        searchSPARQL = re.search("```(.|\n)*```",text)
        if searchSPARQL is None:
            return ''
        start,end = searchSPARQL.span()
        sparql = text[start:end]
        sparql = sparql.replace("```sparql","").replace("```","")
        return sparql


    @staticmethod
    def printAnswer(llmAnswer):
        if llmAnswer['sparql'] is not None:
            finalAnswer = f"""User: {llmAnswer['question']}\nGPT: {llmAnswer['answer']}\n\n\n\nSPARQL:\n{llmAnswer['sparql']}\n-------------------------------------------------------------\n"""
        else:
            finalAnswer =f"""User: {llmAnswer['question']}\nGPT: {llmAnswer['answer']}\n-------------------------------------------------------------\n"""
        print(finalAnswer)