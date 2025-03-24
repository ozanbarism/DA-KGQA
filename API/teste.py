import tiktoken
from configs import *
from core.QuestionHandler_ours import QuestionHandler
from dotenv import load_dotenv
from nlp.normalizer import *
from datetime import datetime
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.ChatHandler import ChatHandler
from core.Configs_loader import load_configs

#OpenAI
load_dotenv()

#KG endpoints and indexes
normalizer, endpoint_t_box, t_box_index, endpoint_a_box, a_box_index = load_configs()

def countTokens(input):
    encoding = tiktoken.encoding_for_model(LLM_MODEL)
    encoded = encoding.encode(input)
    # print(encoded)
    return len(encoded)



questions=["Which <position> is a <Damper Position Command>?",
"What are the instances of damper position commands?",
"Retrieve all damper position commands",
"What commands adjust the position of dampers?",
"List the commands used for damper positioning in this system.",
"What determines how airflows are regulated?"]


questions=["Which <meter> is a <Building Electrical Meter>?",
"What are the instances of building electrical meter?",
"List all building electrical meters",
"Which electrical meters are installed in this building?",
"Show the electrical meters available in this building.",
"What devices are responsible for measuring energy use in this building?",
"Provide details on the equipment used for monitoring energy consumption."]


questions = ["What are the instances of air flow setpoint and the equipment they are a point of?",
"What are the instances of air flow setpoint and their associated equipment?",
"What air flow setpoints exist in this system, and what equipment are they related to?",
"What are the instances of setpoints of air flow and the equipment they are a point of?"]

questions= ["What are the instances of zone air temperature sensors that are points of VAVs that are fed by AHUs?", 
            "Which zone air temperature sensors receive input from terminal units connected to AHUs?", 
            "What are the existing zone air temperature sensors and their associated AHUs?"]

for model in ['lbl/llama','openai/gpt-4o-mini','openai/gpt-4o']:#,]:#'',,'lbl/llama']:#, 'aws/llama-3.1-70b', 'aws/llama-3.1-8b']:
    for question in questions:
        print(f"Question: "+question)
        qa = QuestionHandler(endpoint_a_box,endpoint_t_box,t_box_index,normalizer,a_box_index=a_box_index,model_name=model)
        ttl = qa.getRelevantGraph(question)
        print("ttl:")
        queries,results,selected = qa.textToSPARQL(question,ttl)
        if queries:
            for idx,q in enumerate(queries):
                print("Query "+str(idx)+":")
                print(q)
                print("result:")
                print(str(results[idx]))
                print("----------------------------------")
            print("Query choosed: "+str(selected))
            print(queries[selected])    
            sparql_selected = queries[selected]
            results_selected = results[selected]
            print("Final SPARQL:")
            print(sparql_selected)
            print("Results:")
            print(results_selected)
            print("len",len(results))

#answer = qa.generateNLResponse(question,sparql_selected,results_selected)
#print("answer:")
#print(answer)
